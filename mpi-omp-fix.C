//--------------------------------------------------------------------------------
//  MPI Wrapper
//
//  Author:  Christoph Lehner
//  Year:    2016
//  Summary: Distributes MPI communication over parallel mpi ranks
//--------------------------------------------------------------------------------
// Remark on openMP / openMPI behavior:
//
// This gets full performance if run without any binding but -npernode=X
// Unfortunately, as soon as an opemMP parallel region is used affinity is
// modified which limits MPI performance.  Solution: avoid any openMP parallel
// region for worker ranks while main rank should be fine to use it!
//
// Numerical demonstration of this issue: 
//   knlsubmit01:/root/clehner/shm-nofix-test3-rankfile
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
  MPI_Request request;
};
//--------------------------------------------------------------------------------
#define GET_BLOCK(i) ( (_shm_block*)(blocks_ptr + (block_size + HEADER_SIZE) * (i)) )
#define GET_RECV_BLOCK(i)  GET_BLOCK(2*i + 0)
#define GET_SEND_BLOCK(i)  GET_BLOCK(2*i + 1)
//--------------------------------------------------------------------------------
// Keeping track of Irecvs and Isends
//--------------------------------------------------------------------------------
struct _sendreceive_tag {
  bool send;
  _packet_type p;
  void* buf;
  bool submitted;
  int worker;
};
std::set<_sendreceive_tag*> sendrecv_tags;
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
#define BLOCK_UNROLL 1
#define BLOCK_SIZE (sizeof(float)*16)
//--------------------------------------------------------------------------------
void fast_copy_blocks_threaded(void* dst, const void* src, int nblocks) {

  const float* _src = (float*)src;
  float* _dst = (float*)dst;
  
  int nthreads = omp_get_num_threads();
  int ithread  = omp_get_thread_num();
  for (int i=ithread;i<nblocks;i+=nthreads) {

    const float* _src_ = &_src[i*16];
    float* _dst_ = &_dst[i*16];

    __m512 buffer1 = _mm512_load_ps(_src_);
    _mm512_stream_ps(_dst_, buffer1);

  }

}
//--------------------------------------------------------------------------------
void fast_copy(char* dst, const char* src, size_t size) {

  if (size % 8) {
    fprintf(stderr,"Fast copy only works with size being multiple of 4\n");
    exit(5);
  }

  const double* _src = (double*)src;
  double* _dst = (double*)dst;
  size /= 8;
#pragma omp parallel for
  for (size_t i=0;i<size;i++)
    _dst[i] = _src[i];

#if 0 // actually slower
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
#endif
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
  _shm_block* b = GET_SEND_BLOCK(wid);

  int target = WORKER_ADDR(b->p.node,wid);

  real_MPI_Isend((char*)b + HEADER_SIZE,b->p.size,MPI_CHAR,target,b->p.tag,mpi_world,&b->request);

}
//--------------------------------------------------------------------------------
// Process receive
//--------------------------------------------------------------------------------
static void process_recv(int wid, int bytes, int source, int tag) {

  _shm_block* b = GET_RECV_BLOCK(wid);
  b->p.size = bytes;
  b->p.node = ADDR_NODE(source);
  b->p.tag = tag;

  real_MPI_Irecv((char*)b + HEADER_SIZE,b->p.size,MPI_CHAR,source,b->p.tag,mpi_world,&b->request);
}
//--------------------------------------------------------------------------------
// Worker job
//--------------------------------------------------------------------------------
static int next_send_worker = 0;
static int next_recv_worker = 0;
//--------------------------------------------------------------------------------
static void worker() {
  int worker_id = mpi_id_on_node - 1;

  volatile _shm_block* b_s = GET_SEND_BLOCK(worker_id);
  volatile _shm_block* b_r = GET_RECV_BLOCK(worker_id);

  while (true) {

    if (b_s->status == STATUS_QUIT) {

      b_s->status = STATUS_IDLE;
      b_r->status = STATUS_IDLE;

      break;

    }

    if (b_r->status == STATUS_START) {

      if (!b_r->request) {

	// can receive data from other nodes
	MPI_Status s;
	
	int flag;
	MPI_Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_world, &flag, &s );
	
	if (flag) {

	  int bytes;
	  MPI_Get_count(&s,MPI_CHAR,&bytes);

	  process_recv(worker_id,bytes,s.MPI_SOURCE,s.MPI_TAG);
	  
	}

      } else {

	int flag;
	MPI_Status s;
	MPI_Test((MPI_Request*)&b_r->request,&flag,&s);

	if (flag) {
	  b_r->status = STATUS_IDLE;
	}
      }

    }

    if (b_s->status == STATUS_START) {

      if (!b_s->request) {

	process_send(worker_id);

      } else {

	int flag;
	MPI_Status s;
	MPI_Test((MPI_Request*)&b_s->request,&flag,&s);

	if (flag) {
	  b_s->status = STATUS_IDLE;
	}
      }

    }
  }
}
//--------------------------------------------------------------------------------
static void kill_worker(int wid) {
  _shm_block* b = GET_SEND_BLOCK(wid);
  
  while (b->status != STATUS_IDLE) {
    usleep(0);
  }

  b->status = STATUS_QUIT;
}
//--------------------------------------------------------------------------------
// Barrier
//--------------------------------------------------------------------------------
int MPI_Barrier( MPI_Comm comm ) {
  if (!mpi_init)
    return -1;

  if (comm == mpi_world)
    real_MPI_Barrier(mpi_myrank);
  else
    _printf("Not implemented\n");
  return 0;
}
//--------------------------------------------------------------------------------
static void initiate_sends_and_receives() {

  int cs = 0;
  int cr = 0;

  std::set<_sendreceive_tag*>::iterator r;
  for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {

    _sendreceive_tag* t = *r;
    if (!t->submitted) {

      _shm_block* b;
      if (t->send) {
	b = GET_SEND_BLOCK(t->worker);
	cs++;
      } else {
	b = GET_RECV_BLOCK(t->worker);
	cr++;
      }

      b->request = 0;
      b->status = STATUS_START;
    }

  }

  _printf("STATUS: %d sends, %d receives initiated\n",cs,cr);
}
//--------------------------------------------------------------------------------
static void wait_for_sends_and_receives() {

  std::set<_sendreceive_tag*>::iterator r;
  for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {

    _sendreceive_tag* t = *r;
    if (!t->submitted) {
      _shm_block* b;

      if (t->send) {
	b = GET_SEND_BLOCK(t->worker);
      } else {
	b = GET_RECV_BLOCK(t->worker);
      }

      while (b->status != STATUS_IDLE)
	usleep(0);

    }

  }

}
//--------------------------------------------------------------------------------
static void copy_to_send_buffers() {

  //real_MPI_Barrier(mpi_myrank);

  double size_in_gb = 0.0;

  double t0 = dclock();
  std::set<_sendreceive_tag*>::iterator r;
  for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {
    if ((*r)->submitted == false && (*r)->send) {

      _shm_block* b = GET_SEND_BLOCK((*r)->worker);
      fast_copy((char*)b + HEADER_SIZE,(const char*)(*r)->buf,b->p.size);
      size_in_gb += (double)(*r)->p.size / 1024. / 1024. / 1024.;

    }
  }

  double t1 = dclock();
  _printf("Copy to send buffers %g GB at %g GB/s (%g GB/s memory bandwidth)\n",
	  size_in_gb,size_in_gb/(t1-t0),2.0*size_in_gb/(t1-t0));

  //real_MPI_Barrier(mpi_myrank);

}
//--------------------------------------------------------------------------------
static void copy_from_receive_buffers() {

  // Non-trivial: receive buffers may have become scrambled, need to match

  double size_in_gb = 0.0;
  double t0 = dclock();
  std::set<_sendreceive_tag*>::iterator r;
  for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {
    if ((*r)->submitted == false && !(*r)->send) {

      // need to go through workers and match this receive and copy
      // THAT memory instead of the (*r)->worker one.

      int j;
      for (j=0;j<WORKERS;j++) {
	_shm_block* b = GET_RECV_BLOCK(j);
	if (b->request && b->p.match((*r)->p)) {
	  fast_copy((char*)(*r)->buf,(char*)b + HEADER_SIZE,b->p.size);
	  size_in_gb += (double)(*r)->p.size / 1024. / 1024. / 1024.;
	  break;
	}
      }

      if (j == WORKERS) {
	fprintf(stderr,"Could not match memory block!\n");
	exit(8);
      }

    }
  }

  double t1 = dclock();
  _printf("Copy from receive buffers %g GB at %g GB/s (%g GB/s memory bandwidth)\n",
	  size_in_gb,size_in_gb/(t1-t0),2.0*size_in_gb/(t1-t0));

}
//--------------------------------------------------------------------------------
static void commit_mpi() {

  double size_in_gb = 0.0;

  {
    bool todo = false;
    std::set<_sendreceive_tag*>::iterator r;
    for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {
      if ((*r)->submitted == false) {
	todo = true;
	size_in_gb += (double)(*r)->p.size / 1024. / 1024. / 1024.;
      }
    }

    if (!todo)
      return;
  }

  double t0 = dclock();

  copy_to_send_buffers();

  double t1 = dclock();

  initiate_sends_and_receives();
  
  wait_for_sends_and_receives();

  double t2 = dclock();

  copy_from_receive_buffers();

  double t3 = dclock();

  _printf("TIMING: %g%% in MPI routines (%d sends, %d receives, %g GB at %g GB/s pure MPI would be %g GB/s)\n",
	  100.0*(t2-t1)/(t3-t0),next_send_worker,next_recv_worker,
	  size_in_gb,size_in_gb / (t3 - t0),
	  size_in_gb / (t2-t1));

  {
    std::set<_sendreceive_tag*>::iterator r;
    for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {
      (*r)->submitted = true;
    }
  }

  next_send_worker = 0;
  next_recv_worker = 0;
}
//--------------------------------------------------------------------------------
// MPI_Isend
//--------------------------------------------------------------------------------
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	      MPI_Comm comm, MPI_Request *request) {

  if (next_send_worker >= WORKERS)
    commit_mpi();

  if (comm != mpi_world) {
    fprintf(stderr,"Non-world communicators not yet implemented!\n");
    exit(4);
  }

  if (sizeof(MPI_Request) < sizeof(_sendreceive_tag*)) {
    fprintf(stderr,"MPI_Request has too small size = %d\n",sizeof(MPI_Request));
    exit(1);
  }

  _packet_type p;
  p.tag = tag;
  p.node = dest;
  p.size = count*type_size(datatype);

  if (p.size > block_size) {
    fprintf(stderr,"Block limit currently is %d bytes (tried %d)\n",block_size,p.size);
    exit(7);
  }

  _sendreceive_tag* t = new _sendreceive_tag();
  t->send = true;
  t->p = p;
  t->buf = (void*)buf;
  t->submitted = false;
  t->worker = next_send_worker++;

  GET_SEND_BLOCK(t->worker)->p = p;

  *(_sendreceive_tag**)request = t;

  // dirty hack: test for identical sends
  std::set<_sendreceive_tag*>::iterator r;
  for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {
    if ((*r)->send && (*r)->p.match(p)) {
      fprintf(stderr,"For now overlapping identical sends not allowed!\n");
      exit(12);
    }
  }

  sendrecv_tags.insert(t);

}
//--------------------------------------------------------------------------------
// MPI_Irecv
//--------------------------------------------------------------------------------
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request *request) {

  if (next_recv_worker >= WORKERS)
    commit_mpi();

  if (comm != mpi_world) {
    fprintf(stderr,"Non-world communicators not yet implemented!\n");
    exit(4);
  }

  if (sizeof(MPI_Request) < sizeof(_sendreceive_tag*)) {
    fprintf(stderr,"MPI_Request has too small size = %d\n",sizeof(MPI_Request));
    exit(1);
  }

  _packet_type p;
  p.tag = tag;
  p.node = source;
  p.size = count*type_size(datatype);

  _sendreceive_tag* t = new _sendreceive_tag();
  t->send = false;
  t->p = p;
  t->buf = buf;
  t->submitted = false;
  t->worker = next_recv_worker++;

  GET_RECV_BLOCK(t->worker)->p = p;

  *(_sendreceive_tag**)request = t;

  // dirty hack: test for identical receives
  std::set<_sendreceive_tag*>::iterator r;
  for (r=sendrecv_tags.begin();r!=sendrecv_tags.end();r++) {
    if (!(*r)->send && (*r)->p.match(p)) {
      fprintf(stderr,"For now overlapping identical receives not allowed!\n");
      exit(12);
    }
  }

  sendrecv_tags.insert(t);

  return 0;
}
//--------------------------------------------------------------------------------
// MPI_Wait
//--------------------------------------------------------------------------------
int MPI_Wait(MPI_Request *request, MPI_Status *status) {

  commit_mpi();

  _sendreceive_tag* t = *(_sendreceive_tag**)request;
  sendrecv_tags.erase(t);

  // TODO: fill status of corresponding request
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


  // this is just for debugging purposes:
  //real_MPI_Barrier(mpi_myrank);
  //sleep(5);
  //if (!mpi_node)
  //  printf("--------------------------------------------------------------------------------\n");
  //real_MPI_Barrier(mpi_myrank);
  //

  double t0 = dclock();
  MPI_Request rs, rr;
  MPI_Isend(sendbuf,sendcount,sendtype,dest,sendtag,comm,&rs);
  MPI_Irecv(recvbuf,recvcount,recvtype,source,recvtag,comm,&rr);

  MPI_Status ss;
  MPI_Wait(&rs,&ss);
  MPI_Wait(&rr,status);

  double t1 = dclock();
  double size_in_gb = (double)(sendcount*type_size(sendtype) + recvcount*type_size(recvtype)) / 1024. / 1024. / 1024.;

  _printf("Sendrecv: %g GB at %g GB/s\n",size_in_gb,size_in_gb / (t1-t0));
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
void debug_thread_mapping(bool allthreads) {
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
  
  if (allthreads) {
    if (mpi_id_on_node == 0) {
      // only do this on master node to avoid omp affinity issue
      return;
#pragma omp parallel
      {
	int cpu = sched_getcpu();
	_printf("MPI-FIX:  Rank %d, Thread %d -> Core %d\n",mpi_id,omp_get_thread_num(),core[cpu]);
      }
    }
  } else {
    int cpu = sched_getcpu();
    _printf("MPI-FIX:  Rank %d -> Core %d\n",mpi_id,core[cpu]);
  }
}
//--------------------------------------------------------------------------------
// Startup wrapper
//--------------------------------------------------------------------------------
static int (* real_main)(int argc, char* argv[], char* env[]);
//--------------------------------------------------------------------------------
static int _main(int argc, char* argv[], char* env[]) {

  // init and query mpi
  //real_MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&mpi_thread_provided);
  real_MPI_Init(&argc,&argv);
  mpi_thread_provided = MPI_THREAD_SINGLE;
  real_MPI_Comm_size(mpi_world,&mpi_n);
  real_MPI_Comm_rank(mpi_world,&mpi_id);

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
      _printf("MPI-FIX:  Init %d / %d, Thread-level: %d\n",mpi_id,mpi_n,mpi_thread_provided);

      debug_thread_mapping(false);

      if (verbosity > 1)
	debug_thread_mapping(true);
    }
  }

  // shm parameters
  block_count = WORKERS*2;
  block_size = 16*1024*PAGE_SIZE;

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
