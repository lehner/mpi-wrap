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
#include <set>
#include <sys/file.h>
#include <sys/mman.h>
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
#define MASTER_ADDR(node) ( (node) * rank_per_node )
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
//
// Each worker has a _shm_block
// Even Worker ID uses these to send, odd worker ID to store received data
//
// In case of send operation, the master rank looks for an available 
// worker by checking the status == IDLE.  Since only the master thread
// does this, no mutex is needed.  Then the master thread sets the block
// parameters and sets the status to START.  At this point the worker starts
// sending and at completion returns state to IDLE.
//
// In case of store operation, the worker receives the data and on completion
// sets its status to START.  The master thread checks the readers and if a
// START is found, matches it to the outstanding _receive markers.
// The master thread returns the status to IDLE when data is successfully copied
// to user.
//--------------------------------------------------------------------------------
struct _packet_type {
  int node;
  int tag;
  size_t size;

  bool match(_packet_type& other) {
    return node == other.node && tag == other.tag && size == other.size;
  }
};
//--------------------------------------------------------------------------------
struct _shm_block {
  int status;
#define STATUS_IDLE  0x0
#define STATUS_START 0x1
#define STATUS_QUIT  0x2
  _packet_type p;
};
//--------------------------------------------------------------------------------
// Keeping track of Irecvs
//--------------------------------------------------------------------------------
struct _receive_tag {
  _packet_type p;
  void* buf;
  double t0; // timestamp of submission
};
std::set<_receive_tag*> recv_tags;
//--------------------------------------------------------------------------------
// Dirty hack to avoid out-of-order packets:  Make sure that only a single
// mpi_isend and mpi_irecv is in place with matching _packet_type; stall operation
// to make sure this is correct, or maybe just output error for now and die if
// such a pattern were to be used?
//--------------------------------------------------------------------------------
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
#define GET_BLOCK(i) ( (_shm_block*)(blocks_ptr + (block_size + HEADER_SIZE) * (i)) )
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
// Process send
//--------------------------------------------------------------------------------
static void process_send(int wid) {
  _shm_block* b = GET_BLOCK(wid);

  int target = WORKER_ADDR(b->p.node,wid+1);

  MPI_Send((char*)b + HEADER_SIZE,b->p.size,MPI_CHAR,target,b->p.tag,mpi_world);

  //_printf("Sent data\n");

  b->status = STATUS_IDLE;

}
//--------------------------------------------------------------------------------
static void complete_receive(_shm_block* b, void* buf) {

  double t0 = dclock();
  fast_copy((char*)buf,(char*)b + HEADER_SIZE,b->p.size);
  double t1 = dclock();
  
  //if (verbosity > 1) {
  {
    //double size_in_gb = (double)b->p.size / 1024. / 1024. / 1024.;
    //_printf("Fast-copy (RECV) %g GB/s total %g GB\n",
    //   size_in_gb / (t1-t0), size_in_gb);
  }
  
}
//--------------------------------------------------------------------------------
// Process receive
//--------------------------------------------------------------------------------
static void process_recv(int wid, int bytes, int source, int tag) {

  _shm_block* b = GET_BLOCK(wid);
  b->p.size = bytes;
  b->p.node = ADDR_NODE(source);
  b->p.tag = tag;

  MPI_Status s;
  MPI_Recv((char*)b + HEADER_SIZE,b->p.size,MPI_CHAR,source,b->p.tag,mpi_world,&s);

  //_printf("Received data in block %d\n",wid);

  b->status = STATUS_START;

}
//--------------------------------------------------------------------------------
// Worker job
//--------------------------------------------------------------------------------
static int next_worker_to_use = 0;
//--------------------------------------------------------------------------------
static void worker() {
  int worker_id = mpi_id_on_node - 1;
  bool bsend = (worker_id % 2) == 0;

  //printf("Worker %d has addr %p (%p)\n",worker_id,GET_BLOCK(worker_id),blocks_ptr);

  int last_status = -1;

  while (true) {
    _shm_block* b = GET_BLOCK(worker_id);

    //int status = b->status;
    //if (b->status != last_status) {
    //  _printf("Worker %d status changed to %d (%d)\n",worker_id,b->status,bsend);
    //  last_status = b->status;
    //}

    if (b->status == STATUS_QUIT) {

      b->status = STATUS_IDLE; // TEST
      //_printf("Received quit signal\n");
      break;

    } else if (!bsend && b->status == STATUS_IDLE) {

      // can receive data from other nodes
      MPI_Status s;

      int flag;
      MPI_Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_world, &flag, &s );
      
      if (flag) {

	int bytes;
	MPI_Get_count(&s,MPI_CHAR,&bytes);

	//_printf("Receive size = %d from %d\n",bytes,s.MPI_SOURCE);
	
	if (s.MPI_SOURCE == MASTER_ADDR(mpi_node)) {
	  // special command from master, need to quit only option for now
	  //_printf("Received quit signal from master\n");
	  break;
	}

	process_recv(worker_id,bytes,s.MPI_SOURCE,s.MPI_TAG);
      }

    } else if (bsend && b->status == STATUS_START) {

      //_printf("Start send\n");

      // being told to send data in block cmd
      process_send(worker_id);
    }

  }
}
//--------------------------------------------------------------------------------
static void kill_worker(int wid) {
  _shm_block* b = GET_BLOCK(wid);
  
  bool bsend = (wid % 2) == 0;

  if (bsend) {
    while (b->status != STATUS_IDLE) {
      
      usleep(0);
      
    }

    b->status = STATUS_QUIT;
  } else {
    int t = STATUS_QUIT;
    //_printf("Try killing reader %d\n",wid);
    MPI_Send(&t,1,MPI_INT,WORKER_ADDR(mpi_node,wid),0,mpi_world);
    //_printf("Kill done\n");
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

   
  double _t0 = dclock();

  if (comm != mpi_world) {
    fprintf(stderr,"Non-world communicators not yet implemented!\n");
    exit(4);
  }

  _packet_type p;
  p.tag = tag;
  p.node = dest;
  p.size = count*type_size(datatype);

  if (p.size > block_size) {
    fprintf(stderr,"Block limit currently is %d bytes (tried %d)\n",block_size,p.size);
    exit(7);
  }

  // hack: make sure that no matching send is already in progress
  double t_before_busy = dclock();
  {
    for (int i=0;i<WORKERS;i+=2) {
      _shm_block* b = GET_BLOCK(i);
      while (b->status == STATUS_START && b->p.match(p)) {
	//fprintf(stderr,"For now: do not accept overlapping identical sends\n");
	//exit(5);
	usleep(0);
      }
    }
  }

  // wait for next worker to be available
  _shm_block* b = GET_BLOCK(next_worker_to_use);
  //_printf("Get block %d\n",next_worker_to_use);
  *(_receive_tag**)request = 0;

  next_worker_to_use = (next_worker_to_use + 2) % WORKERS;
  while (b->status != STATUS_IDLE)
    usleep(0);
  double t_after_busy = dclock();

  // OK, go!
  b->p = p;
  double t0 = dclock();
  fast_copy((char*)b + HEADER_SIZE,(const char*)buf,b->p.size);
  double t1 = dclock();
    
  //if (verbosity > 1) {
  double size_in_gb = (double)b->p.size / 1024. / 1024. / 1024.;
  
  //_printf("Set status send %d\n",(int)((long long)b - (long long)blocks_ptr));
  b->status = STATUS_START;

  double _t10 = dclock();
  _printf("SEND %g GB/s total %g GB, %g s (%g%% in fast_copy, %g%% in busy blocks)\n",
	  size_in_gb / (_t10-_t0), size_in_gb, _t10-_t0,
	  (t1-t0) / (_t10-_t0),
	  (t_after_busy - t_before_busy) / (_t10-_t0));
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

  if (sizeof(MPI_Request) < sizeof(_receive_tag*)) {
    fprintf(stderr,"MPI_Request has too small size = %d\n",sizeof(MPI_Request));
    exit(1);
  }

  _packet_type p;
  p.tag = tag;
  p.node = source;
  p.size = count*type_size(datatype);

  _receive_tag* t = new _receive_tag();
  t->p = p;
  t->buf = buf;
  t->t0 = dclock();
  *(_receive_tag**)request = t;

  // dirty hack: test for identical receives
  std::set<_receive_tag*>::iterator r;
  for (r=recv_tags.begin();r!=recv_tags.end();r++) {
    if ((*r)->p.match(p)) {
      fprintf(stderr,"For now overlapping identical receives not allowed!\n");
      exit(12);
    }
  }

  recv_tags.insert(t);

  //_printf("Irecv\n");

  return 0;
}
//--------------------------------------------------------------------------------
// MPI_Wait
//--------------------------------------------------------------------------------
int MPI_Wait(MPI_Request *request, MPI_Status *status) {

  _receive_tag* t = *(_receive_tag**)request;

  if (!t) // no need to wait (send)
    return 0;

  //_printf("WAIT %p\n",request);
  double t0 = dclock();

  while (true) {

    for (int i=1;i<WORKERS;i+=2) {
      _shm_block* b = GET_BLOCK(i);
      if (b->status == STATUS_START && b->p.match(t->p)) {

	double t1 = dclock();

	complete_receive(b,t->buf);
	b->status = STATUS_IDLE;

	recv_tags.erase(t);
	delete t;

	double t2 = dclock();

	double DT = t2 - t->t0;
	double size_in_gb = (double)b->p.size / 1024. / 1024. / 1024.;

	_printf("RECV %g GB/s total %g GB, %g s (%g%% in fast_copy, %g%% in busy blocks)\n",
		size_in_gb / DT, size_in_gb, DT,
		(t2-t1) / DT,
		(t1-t0) / DT);

	//_printf("Receive complete\n");
	return 0;
      }
    }

    usleep(0);
  }

  // never reached
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

  size_t total_size = block_count * (block_size + HEADER_SIZE);

  //_printf("Total-size: %d\n",total_size);

  shm_id = shm_open("/mpi-fix-com",O_CREAT|O_RDWR,0666);
  if (shm_id == -1) {
    fprintf(stderr,"Could not get shared memory with size %d on mpi_id = %d\n",total_size,mpi_id);
    exit(4);
  }

  if (ftruncate(shm_id, total_size) != 0) {
    fprintf(stderr,"Truncate shared memory not successful\n");
    exit(5);
  }

  blocks_ptr = (char*)mmap(NULL, total_size, PROT_READ|PROT_WRITE, MAP_SHARED, shm_id, 0);
  if (MAP_FAILED == (void*)blocks_ptr) {
    fprintf(stderr,"Could not attach to shared memory ID %d with size %d on mpi_id = %d\n",shm_id,
	    total_size,mpi_id);
    exit(4);
  }

  // touch memory
  if (!mpi_id_on_node) {
    memset(blocks_ptr,0,total_size);
  }
    
  // wait until shared memory is initialized
  real_MPI_Barrier(mpi_world);

  if (!mpi_id_on_node)
    *blocks_ptr = 'c';
    
  real_MPI_Barrier(mpi_world);
  
  if (*blocks_ptr != 'c') {
    _printf("Error in shared memory implementation!\n");
    exit(5);
  }

  real_MPI_Barrier(mpi_world);
  
  *blocks_ptr = 0;
  
}
//--------------------------------------------------------------------------------
static void shm_exit() {

  size_t total_size = block_count * (block_size + HEADER_SIZE);
  munmap((void*)blocks_ptr,total_size);
  shm_unlink("/mpi-fix-com");
  
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
  //_printf("Init_thread called\n");
  return 0;
}
//--------------------------------------------------------------------------------
int MPI_Init( int *argc, char ***argv ) {
  //_printf("Init called\n");
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
    kill_worker(i);
  }

  MPI_Comm_free(&mpi_myrank);
  real_MPI_Finalize();
  shm_exit();

  mpi_init = false;
  
  return 0;
}
//--------------------------------------------------------------------------------
