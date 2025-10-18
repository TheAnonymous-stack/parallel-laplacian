#define _GNU_SOURCE
#include "filters.h"
#include <pthread.h>
#include <cstdio>
#include <stdint.h>
#include <sched.h>

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] = {
    0, 1, 0, 1, -4, 1, 0, 1, 0,
};
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] = {
    0, 1, 1, 2, 2, 2,   1,   1,   0, 1, 2, 4, 5, 5,   5,   4,   2,
    1, 1, 4, 5, 3, 0,   3,   5,   4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3,   -12, -24, -12,
    3, 5, 2, 1, 4, 5,   3,   0,   3, 5, 4, 1, 1, 2,   4,   5,   5,
    5, 4, 2, 1, 0, 1,   1,   2,   2, 2, 1, 1, 0,
};
filter log_f = {9, log_m};

/* Identity filter */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

/************** THREAD AFFINITY *****************/
/* Physical core mapping based on lscpu -p output:
 * Cores 0-7: Have hyperthreading (each physical core has 2 logical CPUs)
 * Cores 8-15: No hyperthreading (single logical CPU per physical core)
 * Priority order: Use first logical CPU of each hyperthreaded pair, then non-HT cores
 */
int physical_core_map[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23};

void pin_thread_to_core(int thread_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // Use physical cores in priority order
    int core_id = physical_core_map[thread_id % 16];
    CPU_SET(core_id, &cpuset);
    
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) {
  if (smallest == largest) {
    return;
  }

  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/************** COORDINATE HELPERS *****************/
/* Helper functions for converting coordinates between 2D and 1D system */

int32_t from_1d_to_2d_row(int32_t idx, int32_t num_cols) {
  return idx / num_cols;
}

int32_t from_1d_to_2d_col(int32_t idx, int32_t num_cols) {
  return idx % num_cols;
}

int32_t from_2d_to_1d(int32_t row, int32_t col, int32_t width) {
  return row * width + col;
}

bool is_within_bound(int32_t val, int32_t lower_limit, int32_t upper_limit) {
  return lower_limit <= val && val < upper_limit;
}

int32_t get_max_tiles_horizontal(int32_t chunk_size, int32_t width) {
  int32_t max_tiles_horizontal = (width % chunk_size == 0) ? width / chunk_size : width / chunk_size + 1;
  return max_tiles_horizontal;
}

int32_t get_max_tiles_vertical(int32_t chunk_size, int32_t height) {
  int32_t max_tiles_vertical = (height % chunk_size == 0) ? height / chunk_size : height / chunk_size + 1;
  return max_tiles_vertical;
}

int32_t get_max_tiles(int32_t chunk_size, int32_t width, int32_t height) {
  int32_t max_tiles_horizontal = get_max_tiles_horizontal(chunk_size, width);
  int32_t max_tiles_vertical = get_max_tiles_horizontal(chunk_size, height);
  return max_tiles_horizontal * max_tiles_vertical;
}

// Helper function for each thread to update global max and global min value after applying a filter
void update_smallest_and_largest(common_work* common, int32_t smallest, int32_t largest) {
  pthread_mutex_lock(&(common->lock));
  if (smallest < common->smallest) {
    common->smallest = smallest;
  }
  if (largest > common->largest) {
    common->largest = largest;
  }
  pthread_mutex_unlock(&(common->lock));
}

// Helper to initialize common resource
void init_common_resource(common_work* common_resource, const filter* f, const int32_t* original, int32_t* target, int32_t width, int32_t height, int32_t num_threads) {
  common_resource->f = f;
  common_resource->original_image = original;
  common_resource->output_image = target;
  common_resource->width = width;
  common_resource->height = height;
  common_resource->max_threads = num_threads;
  common_resource->smallest = INT32_MAX;
  common_resource->largest = INT32_MIN;
  pthread_barrier_init(&(common_resource->barrier), NULL, num_threads);
  pthread_mutex_init(&(common_resource->lock), NULL);
}


/*************** COMMON WORK ***********************/
/* Processes a single pixel and returns the value of processed pixel */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
                int32_t width, int32_t height, int row, int column) {
  int32_t filter_size = f->dimension;
  int8_t* filter_matrix = f->matrix;
  
  int32_t half_filter = filter_size / 2;

  // get valid neighborhood bounds
  int32_t start_row = (0 < row - half_filter) ? row - half_filter : 0;
  int32_t end_row = (height > row + half_filter + 1) ? row + half_filter + 1 : height;
  int32_t start_col = (0 < column - half_filter) ? column - half_filter : 0;
  int32_t end_col = (width > column + half_filter + 1) ? column + half_filter + 1 : width;
  
  int32_t sum = 0;
  for (int32_t img_row = start_row; img_row < end_row; img_row++) {
    for(int32_t img_col = start_col; img_col < end_col; img_col++) {
      // get corresponding filter matrix coordinate
      int32_t filter_row = img_row - (row - half_filter);
      int32_t filter_col = img_col - (column - half_filter);

      int32_t img_idx = from_2d_to_1d(img_row, img_col, width);
      int32_t filter_idx = from_2d_to_1d(filter_row, filter_col, filter_size);
      sum += original[img_idx] * filter_matrix[filter_idx];
      
    }
  }
  return sum;
  
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/

void apply_filter2d(const filter *f, const int32_t *original, int32_t *target,
                    int32_t width, int32_t height) {

    int32_t smallest = INT32_MAX;
    int32_t largest = INT32_MIN;

    // filtering each pixel
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int32_t value = apply2d(f, original, target, width, height, i, j);
        target[i * width + j] = value;

        // updating global max and min
        if (value > largest) {
          largest = value;
        }
        if (value < smallest) {
          smallest = value;
        }
      }
    }

    // normalizing filtered pixels
    for (int32_t i = 0; i < height * width; i++) {
      normalize_pixel(target, i, smallest, largest);
    }

}

void *horizontal_sharding(void *arg) {
  work* input = (work*) arg;
  common_work* common = input->common;
  
  // Pin this thread to a physical core
  pin_thread_to_core(input->id);
  
  int32_t nrows = common->height;
  int32_t nthreads = common->max_threads;
  int32_t workload;

  if (nrows % nthreads == 0) {
    workload = nrows / nthreads;
  } else {
    workload = (nrows + nthreads - 1)/nthreads;
  }
  int32_t start = input->id * workload;

  if (start >= nrows) {
    // filter pixel is out of bound at this point
    pthread_barrier_wait(&(common->barrier));
    return NULL;
  }
    
  int32_t end = (start + workload > common->height) ? common->height : start + workload;


  int32_t smallest = INT32_MAX;
  int32_t largest = INT32_MIN;

  for (int32_t i = start; i < end; i++) {
    for (int32_t j = 0; j < common->width; j++) {
      int32_t value = apply2d(common->f, common->original_image, common->output_image, common->width, common->height, i, j);
      (common->output_image)[i * common->width + j] = value;

      // keep track of max and min value seen
      if (value > largest) {
        largest = value;
      }
      if (value < smallest) {
        smallest = value;
      }
    }
  }
  
  update_smallest_and_largest(common, smallest, largest);
  pthread_barrier_wait(&(common->barrier));

  // // normalize pixels
  int32_t global_max = common->largest;
  int32_t global_min = common->smallest;

  for (int32_t i = start; i < end; i++) {
    for (int32_t j = 0; j < common->width; j++) {
      int32_t idx = from_2d_to_1d(i, j, common->width);
      normalize_pixel(common->output_image, idx, global_min, global_max);
      
    }
  }
  return NULL;
}

void*row_major(void* arg) {
  work* input = (work*) arg;
  common_work* common = input->common;
  
  // Pin this thread to a physical core
  pin_thread_to_core(input->id);
  
  int32_t ncols = common->width;
  int32_t nthreads = common->max_threads;
  int32_t workload;

  if (ncols % nthreads == 0) {
    workload = ncols / nthreads;
  } else {
    workload = (ncols + nthreads - 1)/nthreads;
  }

  int32_t start = input->id * workload;

  if (start >= ncols) {
    // filter pixel is out of bound at this point
    pthread_barrier_wait(&(common->barrier));
    return NULL;
  }

  int32_t end = (start + workload > common->width) ? common->width : start + workload;

  int32_t smallest = INT32_MAX;
  int32_t largest = INT32_MIN;

  for (int32_t i = 0; i < common->height; i++) {
    for (int32_t j = start; j < end; j++) {
      int32_t value = apply2d(common->f, common->original_image, common->output_image, common->width, common->height, i, j);
      (common->output_image)[i * common->width + j] = value;
      
      // keep track of max and min value seen
      if (value > largest) {
        largest = value;
      }
      if (value < smallest) {
        smallest = value;
      }

    }
  }
  
  update_smallest_and_largest(common, smallest, largest);
  pthread_barrier_wait(&(common->barrier));

  // normalize pixels
  int32_t global_max = common->largest;
  int32_t global_min = common->smallest;
  for (int32_t i = 0; i < common->height; i++) {
    for (int32_t j = start; j < end; j++) {
      int32_t idx = from_2d_to_1d(i, j, common->width);
      normalize_pixel(common->output_image, idx, global_min, global_max);
    }
  }
  return NULL;
}

void* col_major(void* arg) {
  work* input = (work*) arg;
  common_work* common = input->common;
  
  // Pin this thread to a physical core
  pin_thread_to_core(input->id);
  
  int32_t ncols = common->width;
  int32_t nthreads = common->max_threads;
  int32_t workload;

  if (ncols % nthreads == 0) {
    workload = ncols / nthreads;
  } else {
    workload = (ncols + nthreads - 1)/nthreads;
  }

  int32_t start = input->id * workload;
  
  if (start >= ncols) {
    // filter pixel is out of bound at this point
    pthread_barrier_wait(&(common->barrier));
    return NULL;
  }

  int32_t end = (start + workload > common->width) ? common->width : start + workload;

  int32_t smallest = INT32_MAX;
  int32_t largest = INT32_MIN;

  for (int32_t j = start; j < end; j++) {
    for (int32_t i = 0; i < common->height; i++) {
      int32_t value = apply2d(common->f, common->original_image, common->output_image, common->width, common->height, i, j);
      (common->output_image)[i * common->width + j] = value;

      if (value > largest) {
        largest = value;
      } 
      if (value < smallest) {
        smallest = value;
      }
    }
  }

  update_smallest_and_largest(common, smallest, largest);
  pthread_barrier_wait(&(common->barrier));

  // normalize pixels
  int32_t global_max = common->largest;
  int32_t global_min = common->smallest;

  for (int32_t j = start; j < end; j++) {
    for (int32_t i = 0; i < common->height; i++) {
      int32_t idx = from_2d_to_1d(i, j, common->width);
      normalize_pixel(common->output_image, idx, global_min, global_max);
    }
  }
  return NULL;
}


void* work_queue(void* arg) {
  work* input = (work*) arg;
  common_work* common = input->common;
  queue* q = input->q;
  
  // Pin this thread to a physical core
  pin_thread_to_core(input->id);
  
  int32_t max_tiles_horizontal = get_max_tiles_horizontal(q->chunk_size, common->width);
  int32_t max_tiles_vertical = get_max_tiles_vertical(q->chunk_size, common->height);
  bool isQueueEmpty = false;

  while (!isQueueEmpty) {
    pthread_mutex_lock(&(q->q_lock));
    if (q->next_tile < q->max_tiles) {
      int32_t next_tile = q->next_tile;
      q->next_tile++;
      pthread_mutex_unlock(&(q->q_lock));

      int32_t start_row = next_tile / max_tiles_horizontal * q->chunk_size;
      int32_t end_row = (start_row + q->chunk_size < common->height) ? start_row + q->chunk_size : common->height;
      int32_t start_col = (next_tile % max_tiles_horizontal) * q->chunk_size;
      int32_t end_col = (start_col + q->chunk_size < common->width) ? start_col + q->chunk_size : common->width;

      int32_t largest = INT32_MIN;
      int32_t smallest = INT32_MAX;

      for (int32_t i = start_row; i < end_row; i++) {
        for (int32_t j = start_col; j < end_col; j++) {
          int32_t value = apply2d(common->f, common->original_image, common->output_image, common->width, common->height, i, j);
          (common->output_image)[i * common->width + j] = value;

          if (value > largest) {
            largest = value;
          }
          if (value < smallest) {
            smallest = value;
          }
      }

      update_smallest_and_largest(common, smallest, largest);

      } 
    } else {
        pthread_mutex_unlock(&(q->q_lock));
        isQueueEmpty = true;
    }
  }
  return NULL;
}

// routine function for each thread in work_queue 
// to normalize pixels after the filter has been applied to all tiles
void* queue_normalize_routine(void* arg) {
  work* input  = (work*) arg;
  common_work* common = input->common;
  queue* q = input->q;
  
  // Pin this thread to a physical core
  pin_thread_to_core(input->id);
  int32_t max_tiles_horizontal = get_max_tiles_horizontal(q->chunk_size, common->width);
  int32_t max_tiles_vertical = get_max_tiles_vertical(q->chunk_size, common->height);
  bool isQueueEmpty = false;

  while (!isQueueEmpty) {
    pthread_mutex_lock(&(q->q_lock));
    if (q->next_tile < q->max_tiles) {
      int32_t next_tile = q->next_tile;
      q->next_tile++;
      pthread_mutex_unlock(&(q->q_lock));

      int32_t start_row = next_tile / max_tiles_horizontal * q->chunk_size;
      int32_t end_row = (start_row + q->chunk_size < common->height) ? start_row + q->chunk_size : common->height;
      int32_t start_col = (next_tile % max_tiles_horizontal) * q->chunk_size;
      int32_t end_col = (start_col + q->chunk_size < common->width) ? start_col + q->chunk_size : common->width;

      for (int32_t i = start_row; i < end_row; i++) {
        for (int32_t j = start_col; j < end_col; j++) {
          int32_t idx = from_2d_to_1d(i, j, common->width);
          normalize_pixel(common->output_image, idx, common->smallest, common->largest);
      }

      } 
    } else {
        pthread_mutex_unlock(&(q->q_lock));
        isQueueEmpty = true;
    }
  }

  return NULL;
}

/***************** MULTITHREADED ENTRY POINT ******/
void apply_filter2d_threaded(const filter *f, const int32_t *original,
                             int32_t *target, int32_t width, int32_t height,
                             int32_t num_threads, parallel_method method,
                             int32_t work_chunk) {

  pthread_t threads[num_threads];

  // declare queue
  queue q;
  
  // assign corresponding routine function
  void* (*routine_function) (void *);

  // initialize common resource
  common_work common_resource;
  init_common_resource(&common_resource, f, original, target, width, height, num_threads);

  if (method == SHARDED_ROWS) {
    routine_function = horizontal_sharding;
  }
  else if (method == SHARDED_COLUMNS_COLUMN_MAJOR) {
    routine_function = col_major;
  }
  else if (method == SHARDED_COLUMNS_ROW_MAJOR) {
    routine_function = row_major;
  }
  else if (method == WORK_QUEUE) {
    routine_function = work_queue;
    // initialize queue
    q.chunk_size = work_chunk;
    q.max_tiles = get_max_tiles(work_chunk, width, height);
    q.next_tile = 0;
    pthread_mutex_init(&(q.q_lock), NULL);
  }

  work input_arr[num_threads];
  
  // Pin main thread to core 0 (first in our priority list)
  pin_thread_to_core(0);
  
  // initialize input and create threads
  for (int32_t i = 0; i < num_threads; i++) {
    input_arr[i].common = &common_resource;
    input_arr[i].id = i;
    input_arr[i].q = &q;
    pthread_create(&(threads[i]), NULL, routine_function, &(input_arr[i]));
  }

  // wait for all threads to finish
  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }
  
  if (method == WORK_QUEUE) {

    // compulsory second run to normalize all pixels for work queue
    q.next_tile = 0;
    pthread_t normalize_threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
      pthread_create(&(normalize_threads[i]), NULL, queue_normalize_routine, &(input_arr[i]));
    }

    for (int i = 0; i < num_threads; i++) {
      pthread_join(normalize_threads[i], NULL);
    }
    
  }
}
