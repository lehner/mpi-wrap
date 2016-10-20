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
// Shared memory blocks for each worker
//--------------------------------------------------------------------------------
struct _shm_block {
  int lock;
#define CMD_SEND  0x1
#define CMD_RECV  0x2
  int status;
#define STATUS_IDLE 0x0
#define STATUS_BUSY 0x1
#define STATUS_DONE 0x2
  int count;
  MPI_Datatype type;
  int addr;
  int tag;
  size_t size;
};
#define PAGE_SIZE 4096
#define HEADER_SIZE PAGE_SIZE
static char*  blocks_ptr;
static size_t block_size;
static size_t block_count;
static int shm_id;
//--------------------------------------------------------------------------------
// Worker job
//--------------------------------------------------------------------------------
static void worker() {
  int worker_id = mpi_id_on_node - 1;
  int workers   = rank_per_node - 1;

  int cmd;
  while (1) {
    MPI_Status s;
    MPI_Recv(&cmd,1,MPI_INT,mpi_node_boss_id,0,mpi_world,&s);

    //printf("Command %d received on %d\n",cmd,worker_id);

    if (cmd == -1)
      break;
  }
}
//--------------------------------------------------------------------------------
static void send_to_worker(int worker, int cmd) {
  int wid = mpi_node * rank_per_node + worker + 1;
  MPI_Send(&cmd,1,MPI_INT,wid,0,mpi_world);
}
//--------------------------------------------------------------------------------
// Block management
//--------------------------------------------------------------------------------
static int next_worker = 0;
//--------------------------------------------------------------------------------
static int lock_block() {
  int i;
  for (i=0;i<block_count;i++) {
  _shm_block* b = (_shm_block*)(blocks_ptr + (block_size + HEADER_SIZE) * i);
}
}
//--------------------------------------------------------------------------------
static void release_block(int i) {
}
//--------------------------------------------------------------------------------
static void block_wait(int i) {
}
//--------------------------------------------------------------------------------
static void block_set(int i, int cmd, const void* buf, int count, MPI_Datatype type, int addr, int tag) {
}
//--------------------------------------------------------------------------------
// Barrier
//--------------------------------------------------------------------------------
int MPI_Barrier( MPI_Comm comm ) {
  if (!mpi_init)
    return -1;

  printf("Not implemented\n");
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
  
  int i = lock_block();
  block_set(i, CMD_SEND, buf, count, datatype, dest, tag);
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
  block_wait(i);
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

  size_t total_size = block_count * (block_size + HEADER_SIZE);

  k = ftok(".", 'x');

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

  // touch memory
  memset(blocks_ptr,0,total_size);

}
//--------------------------------------------------------------------------------
static void shm_exit() {

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
    printf("MPI-FIX:  Rank %d, Thread %d -> Core %d\n",mpi_id,omp_get_thread_num(),core[cpu]);
  }
}
//--------------------------------------------------------------------------------
// Startup wrapper
//--------------------------------------------------------------------------------
extern "C" int __libc_start_main(void* func_ptr,
				 int argc,
				 char* argv[],
				 void (*init_func)(void),
				 void (*fini_func)(void),
				 void (*rtld_fini_func)(void),
				 void *stack_end) {
  

#define LOAD(s) *((void**)&real_ ## s) = dlsym(RTLD_NEXT, #s); if (!real_ ## s) { fprintf(stderr,"MPI-OMP-FIX: Could not find %s\n", #s); return false; }

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

  // debug output
  if (verbosity > 0) {
    
    if (!mpi_node) {
      printf("MPI-FIX:  Init %d / %d, Thread-level: %d, OMP Threads %d\n",mpi_id,mpi_n,mpi_thread_provided,nthreads);

      if (verbosity > 1)
	debug_thread_mapping();
    }
  }

  // shm parameters
  block_count = (rank_per_node - 1) * 2;
  block_size = 1024*PAGE_SIZE;

  // create shared memory
  shm_init();

  if (!mpi_id_on_node) {
    return real___libc_start_main(func_ptr,argc,argv,init_func,fini_func,rtld_fini_func,stack_end);
  } else {
    worker();
    if (verbosity > 0 && !mpi_node)
      printf("MPI-FIX:  Exit %d / %d\n",mpi_id,mpi_n);

    real_MPI_Finalize();
    shm_exit();
    exit(0); // needed since __libc_start_main is never called
    return 0; // no warning
  }
}
//--------------------------------------------------------------------------------
// Trivial functions
//--------------------------------------------------------------------------------
int MPI_Init_thread( int *argc, char ***argv, int required, int *provided ) {
  if (provided)
    *provided = mpi_thread_provided;
  return 0;
}
//--------------------------------------------------------------------------------
int MPI_Init( int *argc, char ***argv ) {
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

  int i;
  for (i=0;i<rank_per_node - 1;i++)
    send_to_worker(i,-1); // tell worker to quit
  
  real_MPI_Finalize();
  shm_exit();

  mpi_init = false;
  
  return 0;
}
//--------------------------------------------------------------------------------
