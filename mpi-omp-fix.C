//--------------------------------------------------------------------------------
//  MPI Wrapper
//
//  Author:  Christoph Lehner
//  Year:    2016
//  Summary: Distributes MPI communication over parallel mpi ranks
//--------------------------------------------------------------------------------
#include <mpi.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <ctime>
#include <unistd.h>
#include "immintrin.h"
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <pthread.h>
//--------------------------------------------------------------------------------
// We implement the following functions:
//--------------------------------------------------------------------------------
int (* real_MPI_Init_thread)( int *argc, char ***argv, int required, int *provided );
int (* real_MPI_Init)( int *argc, char ***argv );
int (* real_MPI_Barrier)( MPI_Comm comm );
int (* real_MPI_Sendrecv)(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
			  int dest, int sendtag,
			  void *recvbuf, int recvcount, MPI_Datatype recvtype,
			  int source, int recvtag,
			  MPI_Comm comm, MPI_Status *status);
int (* real_MPI_Isend)(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
		       MPI_Comm comm, MPI_Request *request);
int (* real_MPI_Irecv)(void *buf, int count, MPI_Datatype datatype, int source,
		       int tag, MPI_Comm comm, MPI_Request *request);
int (* real_MPI_Wait)(MPI_Request *request, MPI_Status *status);
int (* real_MPI_Comm_size)( MPI_Comm comm, int *size );
int (* real_MPI_Comm_rank)(MPI_Comm comm, int *rank);
int (* real_MPI_Finalize)( void );
int (* real___libc_start_main)(void *func_ptr,
			       int argc,
			       char* argv[],
			       void (*init_func)(void),
			       void (*fini_func)(void),
			       void (*rtld_fini_func)(void),
			       void *stack_end);
//--------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------
static MPI_Comm mpi_world;
static MPI_Comm mpi_myrank;
static int mpi_thread_provided;
static int mpi_id;
static int mpi_n;
static int mpi_node;
static int mpi_nodes;
static int mpi_id_on_node;
static int verbosity;
static int nthreads;
static int rank_per_node;
static int mpi_node_boss_id;
static bool mpi_init;
//--------------------------------------------------------------------------------
// Mapping of addresses
//
// Idea: 
// - Rank 0 runs MPI program
// - Rank i runs data channels
//--------------------------------------------------------------------------------
#define WORKER_ADDR(node,worker) ( (node) * rank_per_node + 1 + (worker))
#define WORKERS ( rank_per_node - 1 )
//--------------------------------------------------------------------------------
static void _printf(const char* format, ...) {
  va_list args;
  va_start(args, format);

  char buf[2048];
  vsprintf(buf,format, args);
  printf("[%d,%d]  %s",mpi_node,mpi_id_on_node,buf);
}
//--------------------------------------------------------------------------------
// Shared memory blocks
// These blocks hold data to be sent by the workers (CMD_SEND) as
// well as data received from another node (CMD_STORE).  They also
// mark data that is expected to be received (CMD_RECV).
//--------------------------------------------------------------------------------
struct _mutex {
  pthread_mutex_t lock_mutex;
  pthread_mutexattr_t lock_mutex_attr;
};
//--------------------------------------------------------------------------------
struct _worker_cmd {
  //_mutex l; // don't need a mutex since only one thread sends commands to me and
  // only the worker itself can return to the idle state
  int cmd;
};
//--------------------------------------------------------------------------------
struct _shm_block {
  _mutex l;
  int lock;
  int cmd;
#define CMD_SEND  0x1
#define CMD_STORE  0x2
#define CMD_RECV 0x3
  int status;
#define STATUS_IDLE  0x0
#define STATUS_START 0x1
#define STATUS_BUSY  0x2
#define STATUS_DONE  0x3
  int count;
  MPI_Datatype type;
  int addr;
  int tag;
  size_t size;
  int worker;
  void* head_addr;
  int block_forward;
};
#define PAGE_SIZE 4096
#define HEADER_SIZE PAGE_SIZE
#define WORKER_COMM_SIZE PAGE_SIZE*2
static char*  blocks_ptr;
static size_t block_size;
static size_t block_count;
static int shm_id;
//--------------------------------------------------------------------------------
// Benchmark
//--------------------------------------------------------------------------------
inline double dclock() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (1.0*tv.tv_usec + 1.0e6*tv.tv_sec) / 1.0e6;
}
//--------------------------------------------------------------------------------
// Fast copy
//--------------------------------------------------------------------------------
#define BLOCK_SIZE (sizeof(float)*16)
#define GET_BLOCK(i) ( (_shm_block*)(blocks_ptr + WORKER_COMM_SIZE + (block_size + HEADER_SIZE) * (i)) )
#define GET_WORKER_CMD(i) ( (_worker_cmd*)(blocks_ptr + sizeof(_worker_cmd)*(i) ) )
//--------------------------------------------------------------------------------
void fast_copy_blocks_threaded(void* dst, const void* src, int nblocks) {

  const float* _src = (float*)src;
  float* _dst = (float*)dst;
  
  int nthreads = omp_get_num_threads();
  int ithread  = omp_get_thread_num();
  for (int i=ithread;i<nblocks;i+=nthreads) {
    __m512 buffer = _mm512_load_ps(_src + i*16);
    _mm512_stream_ps(_dst + i*16 , buffer);
  }

}
//--------------------------------------------------------------------------------
void fast_copy(char* dst, const char* src, size_t size) {

  size_t nfastblocks = (size - size % BLOCK_SIZE) / BLOCK_SIZE;
  size_t size_slow = size - nfastblocks * BLOCK_SIZE;

  // start copy threads
#pragma omp parallel
  {
    fast_copy_blocks_threaded(dst,src,nfastblocks);
  }

  if (size_slow)
    memcpy(dst + nfastblocks * BLOCK_SIZE,
	   src + nfastblocks * BLOCK_SIZE, 
	   size_slow);
}
//--------------------------------------------------------------------------------
// Block management
//--------------------------------------------------------------------------------
#define ADDR_NODE(addr)  ( (int)( addr / rank_per_node ) )
//--------------------------------------------------------------------------------
static int type_size(MPI_Datatype t) {

  if (t == MPI_CHAR) {
    return 1;
  } else if (t==MPI_DOUBLE) {
    return 8;
  } else if (t==MPI_FLOAT) {
    return 4;
  } else if (t==MPI_INT) {
    return 4;
  } else {
    fprintf(stderr,"Unknown data-type: %d\n",*(int*)&t);
    exit(5);
  }
}
//--------------------------------------------------------------------------------
static bool match_block_metadata(_shm_block* A, _shm_block* B) {
  return A->tag == B->tag &&
    A->size == B->size &&
    ADDR_NODE(A->addr) == ADDR_NODE(B->addr);
}
//--------------------------------------------------------------------------------
// Process send
//--------------------------------------------------------------------------------
static void process_send(int i) {
  _shm_block* b = GET_BLOCK(i);

  if (b->cmd == CMD_SEND) {

    //_printf("CMD Send to %d with tag %d\n",b->addr,b->tag);
    b->status = STATUS_BUSY;
    MPI_Send((char*)b + HEADER_SIZE,b->count,b->type,b->addr,b->tag,mpi_world);
    b->status = STATUS_DONE;

    _printf("Sent data\n");
    //_printf("CMD Send done\n");

  } else {

    _printf("Unknown command %d!\n",b->cmd);

  }

}
//--------------------------------------------------------------------------------
// Lock a block and assign a worker to it
//--------------------------------------------------------------------------------
#define LOCK(l)					\
  pthread_mutex_lock(&l.lock_mutex);
#define UNLOCK(l)				\
  pthread_mutex_unlock(&l.lock_mutex);
//--------------------------------------------------------------------------------
static int lock_block() {
  int i;

  while (true) {
    for (i=0;i<block_count;i++) {
      _shm_block* b = GET_BLOCK(i);

      LOCK(b->l);
      if (!b->lock) {
	b->lock = 1;
	b->worker = i;
	if (i >= WORKERS) {
	  fprintf(stderr,"Logic error in lock_block, block_count and WORKERS not correctly related\n");
	  exit(44);
	}
	

	UNLOCK(b->l);
	//_printf("lock_block(block = %d, worker = %d)\n",i,b->worker);
	return i;
      }
      UNLOCK(b->l);
    }

    _printf("Warning: lock_block failed\n");
    sleep(1);
  }

  return -1; // no warning
}
//--------------------------------------------------------------------------------
static void release_block(int i) {
  _shm_block* b = GET_BLOCK(i);
  LOCK(b->l);
  if (!b->lock) {
    fprintf(stderr,"Logic error in release_block\n");
    exit(5);
  }
  //_printf("release_block(%d)\n",i);
  b->lock = 0;
  UNLOCK(b->l);
}
//--------------------------------------------------------------------------------
static void complete_receive(_shm_block* b, _shm_block* bp) {

  double t0 = dclock();
  fast_copy((char*)b->head_addr,(char*)bp + HEADER_SIZE,b->size);
  double t1 = dclock();
  
  //if (verbosity > 1) {
  {
    double size_in_gb = (double)b->size / 1024. / 1024. / 1024.;
    //_printf("Fast-copy (RECV) %g GB/s total %g GB\n",
    //   size_in_gb / (t1-t0), size_in_gb);
  }
  
}
//--------------------------------------------------------------------------------
static void block_wait(int i) {
  _shm_block* b = GET_BLOCK(i);

  while (b->status != STATUS_DONE) {
    usleep(0);

    // if I should read data, see if it is already here in any of the blocks
    if (b->cmd == CMD_RECV) {
      
      int j;
      for (j=0;j<block_count;j++) {
	_shm_block* bp = GET_BLOCK(j);

	//_printf("Block %d %d %d\n",j,bp->lock,bp->cmd);

	/*if (bp->lock && bp->cmd==CMD_STORE) {
	  _printf("Test block: STORE tag(%d,%d), addr(%d,%d) size(%d,%d)\n",b->tag,bp->tag,ADDR_NODE(b->addr),ADDR_NODE(bp->addr),
		  b->size,bp->size);
		  }*/

	if (bp->lock && bp->cmd==CMD_STORE &&
	    match_block_metadata(b,bp)) {

	  //_printf("Found matching CMD_STORE!\n");

	  complete_receive(b,bp);

	  release_block(j); // release the store block
	  return;
	}
      }      
    }

    //_printf("Wait Block %d (%d,%d)\n",i,b->cmd,b->status);
  }

  if (b->cmd == CMD_RECV) {
    // this can happen if worker found CMD_RECV waiting and directly copied
    // results to this block
    complete_receive(b,b);
  }

  b->status = STATUS_IDLE;
}
//--------------------------------------------------------------------------------
// Process receive
//--------------------------------------------------------------------------------
static void process_recv(int bytes, int source, int tag) {

  _shm_block* b;

  // variant 1) find matching receive block
  int j;
  for (j=0;j<block_count;j++) {
    _shm_block* bp = GET_BLOCK(j);

    LOCK(bp->l);
    if (bp->lock && bp->status == STATUS_START &&
	bp->cmd==CMD_RECV &&
	bp->size == bytes && 
	ADDR_NODE(source) == ADDR_NODE(bp->addr) &&
	bp->tag == tag) {

      bp->status = STATUS_BUSY;
      UNLOCK(bp->l);
      //_printf("FOUND WAITING\n");
      b = bp;
      break;
    }
    UNLOCK(bp->l);
  }

  if (j == block_count) {
    j = lock_block();
    
    b = GET_BLOCK(j);
    b->addr = source;
    b->tag = tag;
    b->size = bytes;
    b->cmd = CMD_STORE;

    //_printf("CMD Store from %d with tag %d\n",b->addr,b->tag);
  }

  b->status = STATUS_BUSY;
  MPI_Status s;
  MPI_Recv((char*)b + HEADER_SIZE,b->size,MPI_CHAR,source,b->tag,mpi_world,&s);
  _printf("Received data in block %d (%d)\n",j,b->cmd);
  b->status = STATUS_DONE;

}
//--------------------------------------------------------------------------------
// Worker job
//--------------------------------------------------------------------------------
#define WORKER_CMD_QUIT -1
#define WORKER_CMD_IDLE -2
//--------------------------------------------------------------------------------
static int next_worker_to_use = 0;
//--------------------------------------------------------------------------------
static void worker() {
  int worker_id = mpi_id_on_node - 1;
  _worker_cmd* wc = GET_WORKER_CMD(worker_id);
      
  int cmd;
  while (1) {

    cmd = wc->cmd;

    if (cmd == WORKER_CMD_QUIT) {
      _printf("Received quit signal\n");
      break;
    } else if (cmd == WORKER_CMD_IDLE) {

      // can receive data from other nodes
      MPI_Status s;

      int flag;
      MPI_Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_world, &flag, &s );
      
      if (flag) {
	
	int bytes;
	MPI_Get_count(&s,MPI_CHAR,&bytes);

	process_recv(bytes,s.MPI_SOURCE,s.MPI_TAG);
      }

    } else {

      // being told to send data in block cmd
      process_send(cmd);

      wc->cmd = WORKER_CMD_IDLE;

    }

  }
}
//--------------------------------------------------------------------------------
void worker_send_cmd(int wid, int cmd) {
  _worker_cmd* wc = GET_WORKER_CMD(wid);
  
  while (true) {

    if (wc->cmd == WORKER_CMD_IDLE) {
      wc->cmd = cmd;
      return;
    }

    usleep(0);

    // STALL HERE
    //_printf("worker %d stalling before cmd\n",wid,cmd);

  }

}
//--------------------------------------------------------------------------------
// Send to target node that worker node will transfer data, learn which target
// address to use
//--------------------------------------------------------------------------------
static void block_set(int i, int cmd, void* buf, int count, MPI_Datatype type, int node, int tag) {
  _shm_block* b = GET_BLOCK(i);

  if (!b->lock) {
    fprintf(stderr,"Logic error in block_set\n");
    exit(5);
  }

  b->cmd = cmd;
  b->status = STATUS_START;
  b->count = count;
  b->type = type;
  b->tag = tag;

  b->size = count*type_size(type);

  if (b->size > block_size) {
    fprintf(stderr,"Block limit currently is %d bytes (tried %d)\n",block_size,b->size);
    exit(7);
  }

  if (b->cmd == CMD_SEND) {

    //_printf("CMD_SEND\n");

    double t0 = dclock();
    fast_copy((char*)b + HEADER_SIZE,(const char*)buf,b->size);
    double t1 = dclock();
    
    //if (verbosity > 1) {
    {
      double size_in_gb = (double)b->size / 1024. / 1024. / 1024.;
      //_printf("SEND Fast-copy %g GB/s total %g GB for worker %d, target %d\n",
      //   size_in_gb / (t1-t0), size_in_gb,b->worker,node);
    }

    b->worker = next_worker_to_use;
    b->addr = WORKER_ADDR(node,b->worker + 1); // +1 to get the corresponding recv worker

    worker_send_cmd(b->worker,i);

    // only use even workers for send, odd to receive
    next_worker_to_use = (next_worker_to_use + 2) % WORKERS;

    //_printf("SEND Sent send command to worker %d\n",b->worker);

  } else if (b->cmd == CMD_RECV) {
    b->head_addr = buf; // remember where to copy stuff
    b->addr = node * rank_per_node;
  }
  
}
//--------------------------------------------------------------------------------
// Barrier
//--------------------------------------------------------------------------------
int MPI_Barrier( MPI_Comm comm ) {
  if (!mpi_init)
    return -1;

  _printf("Not implemented\n");
  return 0;
}
//--------------------------------------------------------------------------------
// MPI_Isend
//--------------------------------------------------------------------------------
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	      MPI_Comm comm, MPI_Request *request) {

   
  if (comm != mpi_world) {
    fprintf(stderr,"Non-world communicators not yet implemented!\n");
    exit(4);
  }

  //_printf("Node %d MPI_Isend(dest = %d,tag = %d)\n",mpi_node,dest,tag);

  int i = lock_block();
  block_set(i, CMD_SEND, (void*)buf, count, datatype, dest, tag);
  *(int*)request = i;

  return 0;
}
//--------------------------------------------------------------------------------
// MPI_Irecv
//--------------------------------------------------------------------------------
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request *request) {

  if (comm != mpi_world) {
    fprintf(stderr,"Non-world communicators not yet implemented!\n");
    exit(4);
  }
  
  //_printf("Node %d MPI_Irecv(source = %d,tag = %d)\n",mpi_node,source,tag);

  int i = lock_block();
  block_set(i, CMD_RECV, buf, count, datatype, source, tag);
  *(int*)request = i;

  return 0;
}
//--------------------------------------------------------------------------------
// MPI_Wait
//--------------------------------------------------------------------------------
int MPI_Wait(MPI_Request *request, MPI_Status *status) {



  int i = *(int*)request;

  _printf("MPI_Wait start (%d:%d)\n",i,GET_BLOCK(i)->cmd);

  double t0 = dclock();
  block_wait(i);

  double t1 = dclock();
  _printf("MPI_Wait complete (%d:%d, t=%g)\n",i,GET_BLOCK(i)->cmd,t1-t0);
  release_block(i);

  return 0;
}
//--------------------------------------------------------------------------------
// Sendrecv
//--------------------------------------------------------------------------------
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		 int dest, int sendtag,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype,
		 int source, int recvtag,
		 MPI_Comm comm, MPI_Status *status) {

  if (!mpi_init)
    return -1;

  MPI_Request rs, rr;
  MPI_Isend(sendbuf,sendcount,sendtype,dest,sendtag,comm,&rs);
  MPI_Irecv(recvbuf,recvcount,recvtype,source,recvtag,comm,&rr);

  MPI_Status ss;
  MPI_Wait(&rs,&ss);
  MPI_Wait(&rr,status);

  return 0;
}
//--------------------------------------------------------------------------------
// Shared memory initialization / destruction
//--------------------------------------------------------------------------------
static void shm_init() {

  key_t          k;

  size_t total_size = block_count * (block_size + HEADER_SIZE) + WORKER_COMM_SIZE;

  k = 0x7f7d + mpi_node; //ftok(".", 'x');

  if (!mpi_id_on_node) {

    shm_id = shmget(k, total_size, IPC_CREAT | 0666);

    real_MPI_Barrier(mpi_world);

  } else {
    real_MPI_Barrier(mpi_world);

    shm_id = shmget(k, total_size, 0666);
  }

  if (shm_id < 0) {
    fprintf(stderr,"Could not get shared memory key %X with size %d on mpi_id = %d\n",k,total_size,mpi_id);
    exit(4);
  }

  blocks_ptr = (char*)shmat(shm_id, NULL, 0);
  if ((void *) -1 == blocks_ptr) {
    fprintf(stderr,"Could not attach to shared memory key %X (ID %d) with size %d on mpi_id = %d\n",k,shm_id,
	    total_size,mpi_id);
    exit(4);
  }
  
#define LOCK_CREATE(l)							\
  pthread_mutexattr_init(&l.lock_mutex_attr);				\
  pthread_mutexattr_setpshared(&l.lock_mutex_attr, PTHREAD_PROCESS_SHARED); \
  pthread_mutex_init(&l.lock_mutex,&l.lock_mutex_attr);

#define LOCK_DESTROY(l)					\
  pthread_mutex_destroy(&l.lock_mutex);			\
  pthread_mutexattr_destroy(&l.lock_mutex_attr); 

  // touch memory
  //_printf("Before\n");
  if (!mpi_id_on_node) {
    memset(blocks_ptr,0,total_size);
    int i;
    for (i=0;i<block_count;i++) {
      _shm_block* b = GET_BLOCK(i);
      LOCK_CREATE(b->l);
    }
    for (i=0;i<WORKERS;i++) {
      _worker_cmd* wc = GET_WORKER_CMD(i);
      wc->cmd = WORKER_CMD_IDLE;
    }
  }
  //_printf("After\n");
  
  // wait until shared memory is initialized
  real_MPI_Barrier(mpi_world);
}
//--------------------------------------------------------------------------------
static void shm_exit() {

  if (!mpi_id_on_node) {
    int i;
    for (i=0;i<block_count;i++) {
      _shm_block* b = GET_BLOCK(i);
      LOCK_DESTROY(b->l);
    }
  }

  shmdt((void *)blocks_ptr);
  if (!mpi_id_on_node)
    shmctl(shm_id, IPC_RMID, NULL);

}
//--------------------------------------------------------------------------------
// Debug function to print the thread mapping
//--------------------------------------------------------------------------------
void debug_thread_mapping() {
  std::vector<int> core;
  {
    FILE* f = fopen("/proc/cpuinfo","rt");
    while (!feof(f)) {
      char buf[1024];
      fgets(buf,1023,f);
      int c;
      if (sscanf(buf,"core id		: %d",&c)==1)
	core.push_back(c);
    }
    fclose(f);
  }
  
#pragma omp parallel
  {
    int cpu = sched_getcpu();
    _printf("MPI-FIX:  Rank %d, Thread %d -> Core %d\n",mpi_id,omp_get_thread_num(),core[cpu]);
  }
}
//--------------------------------------------------------------------------------
// Startup wrapper
//--------------------------------------------------------------------------------
static int (* real_main)(int argc, char* argv[], char* env[]);
//--------------------------------------------------------------------------------
static int _main(int argc, char* argv[], char* env[]) {

  // init and query mpi
  real_MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&mpi_thread_provided);
  real_MPI_Comm_size(mpi_world,&mpi_n);
  real_MPI_Comm_rank(mpi_world,&mpi_id);

  // query openmp
#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
    }
  }

  _printf("Running on %d threads\n",nthreads);

  // test environment
  {
    char* t;

    t = getenv("MPI_FIX_RANK_PER_NODE");
    if (!t) {
      if (!mpi_id)
	fprintf(stderr,"Could not find MPI_FIX_RANK_PER_NODE environment variable.\n");
      exit(2);
    }

    rank_per_node = atoi(t);

    t = getenv("MPI_FIX_VERBOSITY");
    if (t)
      verbosity = atoi(t);
    else
      verbosity = 0;
    
  }

  // get coarse grid coordinates
  if (mpi_n % rank_per_node) {
    fprintf(stderr,"MPI_FIX_RANK_PER_NODE(%d) does not divide total world size(%d)\n",rank_per_node,mpi_n);
    exit(3);
  }

  mpi_nodes = mpi_n / rank_per_node;
  mpi_node = mpi_id / rank_per_node;
  mpi_id_on_node = mpi_id % rank_per_node;
  mpi_node_boss_id = mpi_node * rank_per_node;

  // init my rank communicator
  MPI_Comm_split(mpi_world,mpi_id_on_node,mpi_id,&mpi_myrank);

  // debug output
  if (verbosity > 0) {
    
    if (!mpi_node) {
      _printf("MPI-FIX:  Init %d / %d, Thread-level: %d, OMP Threads %d\n",mpi_id,mpi_n,mpi_thread_provided,nthreads);

      if (verbosity > 1)
	debug_thread_mapping();
    }
  }

  // shm parameters
  block_count = WORKERS;
  block_size = 16*1024*PAGE_SIZE;

  // need an even number of workers (one for send, one for receive)
  if (WORKERS % 2) {
    fprintf(stderr,"Need even number of workers (workers = %d)\n",WORKERS);
    exit(4);
  }

  // create shared memory
  shm_init();

  if (!mpi_id_on_node) {
    return real_main(argc,argv,env);
  } else {
    worker();
    if (verbosity > 0 && !mpi_node)
      _printf("MPI-FIX:  Exit %d / %d\n",mpi_id,mpi_n);

    MPI_Comm_free(&mpi_myrank);
    real_MPI_Finalize();
    shm_exit();
    exit(0); // needed since __libc_start_main is never called
    return 0; // no warning
  }

}
//--------------------------------------------------------------------------------
extern "C" int __libc_start_main(int (* func_ptr)(int argc, char* argv[], char* env[]),
				 int argc,
				 char* argv[],
				 void (*init_func)(void),
				 void (*fini_func)(void),
				 void (*rtld_fini_func)(void),
				 void *stack_end) {
  

#define LOAD(s) *((void**)&real_ ## s) = dlsym(RTLD_NEXT, #s); if (!real_ ## s) { fprintf(stderr,"MPI-OMP-FIX: Could not find %s\n", #s); return false; }

  // real main
  real_main = func_ptr;

  // load real implementations
  LOAD(MPI_Init_thread);
  LOAD(MPI_Init);
  LOAD(MPI_Isend);
  LOAD(MPI_Irecv);
  LOAD(MPI_Wait);
  LOAD(MPI_Barrier);
  LOAD(MPI_Sendrecv);
  LOAD(MPI_Comm_size);
  LOAD(MPI_Comm_rank);
  LOAD(MPI_Finalize);
  LOAD(__libc_start_main);

  // defaults
  mpi_init = true;

  // get communicators
#ifdef OMPI_MPI_COUNT_TYPE
  // have openmpi
  {
    mpi_world = (MPI_Comm)dlsym(RTLD_DEFAULT, "ompi_mpi_comm_world");
    if (!mpi_world) {
      fprintf(stderr,"Could not locate openmpi world communicator\n");
      exit(1);
    }
  }
#else
  mpi_world = MPI_COMM_WORLD;
#endif

  return real___libc_start_main((void*)_main,argc,argv,init_func,fini_func,rtld_fini_func,stack_end);
}
//--------------------------------------------------------------------------------
// Trivial functions
//--------------------------------------------------------------------------------
int MPI_Init_thread( int *argc, char ***argv, int required, int *provided ) {
  if (provided)
    *provided = mpi_thread_provided;
  _printf("Init_thread called\n");
  return 0;
}
//--------------------------------------------------------------------------------
int MPI_Init( int *argc, char ***argv ) {
  _printf("Init called\n");
  return 0;
}
//--------------------------------------------------------------------------------
int MPI_Comm_size( MPI_Comm comm, int *size ) { 
  *size = mpi_nodes;
  return 0;
}
//--------------------------------------------------------------------------------
int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  *rank = mpi_node;
  return 0;
}
//--------------------------------------------------------------------------------
int MPI_Finalize( void ) {

  if (!mpi_init)
    return -1;

  real_MPI_Barrier(mpi_myrank);

  int i;
  for (i=0;i<WORKERS;i++) {
    worker_send_cmd(i,WORKER_CMD_QUIT);
  }

  MPI_Comm_free(&mpi_myrank);
  real_MPI_Finalize();
  shm_exit();

  mpi_init = false;
  
  return 0;
}
//--------------------------------------------------------------------------------
