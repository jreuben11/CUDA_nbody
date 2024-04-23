
# [01-nbody.cu](01-nbody.cu) - CPU baseline
```bash
nvcc -std=c++11 -o nbody-01 01-nbody.cu
./nbody-01
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody-01 
```

# [02-nbody.cu](02-nbody.cu) - convert to kernel with error handling
```C
#include <assert.h>
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// void bodyForce(Body *p, float dt, int n) {
__global__ void bodyForce(Body *p, float dt, int n) {

cudaError_t mallocErr, syncErr, asyncErr;
// buf = (float *)malloc(bytes); 
mallocErr = cudaMallocManaged(&buf, bytes);
checkCuda(mallocErr);

// bodyForce(p, dt, nBodies); 
int numThreads = 1;
int numBlocks = 1;
bodyForce<<<numBlocks, numThreads>>>(p, dt, nBodies); 
syncErr = cudaGetLastError();
asyncErr = cudaDeviceSynchronize();
if (syncErr != cudaSuccess) printf("Sync Error: %s\n", cudaGetErrorString(syncErr));
if (asyncErr != cudaSuccess) printf("Async Error: %s\n", cudaGetErrorString(asyncErr));

// free(buf);
cudaFree(buf);
```

```bash
nvcc -std=c++11 -o nbody-02 02-nbody.cu
./nbody-02
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody-02
```


# [03-nbody.cu](03-nbody.cu) - thread stride
```C
 // KERNEL
  // for (int i = 0; i < n; ++i) {
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride) {

// int numThreads = 1;
int numThreads = 256;
```

```bash
nvcc -std=c++11 -o nbody-03 03-nbody.cu
./nbody-03
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody-03
```

# [04-nbody.cu](04-nbody.cu) - grid-stride loop
```C
// KERNEL
  // int index = threadIdx.x;
  // int stride = blockDim.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;


// int numBlocks = 1;
int blockSize = 256;
int numBlocks = (nBodies + blockSize - 1) / blockSize;

```
```bash
nvcc -std=c++11 -o nbody-04 04-nbody.cu
./nbody-04
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody-04
```

# [05-nbody.cu](05-nbody.cu) - query device for SM -> Blocksize, cudaMemPrefetchAsync
```C
int deviceId;
int numberOfSMs;
cudaGetDevice(&deviceId);
cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

// numBlocks =  numberOfSMs; // UNSURE WHY BLOCK > 15 WAS NOT AVAILABLE IN KERNEL

for (int iter = 0; iter < nIters; iter++) {
  // ADD cudaMemPrefetchAsync to GPU
  cudaMemPrefetchAsync(buf, bytes, deviceId);
  asyncErr = cudaDeviceSynchronize();
  checkCuda(asyncErr);
  // ADD cudaMemPrefetchAsync

  bodyForce<<<numBlocks, numThreads>>>(p, dt, nBodies); 
  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) printf("Sync Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Async Error: %s\n", cudaGetErrorString(asyncErr));

  // ADD cudaMemPrefetchAsync to CPU
  cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId); // TEMPORARY until we move this to GPU !
  // ADD cudaMemPrefetchAsync
  for (int i = 0 ; i < nBodies; i++) { // integrate position

```
```bash
nvcc -std=c++11 -o nbody-05 05-nbody.cu
./nbody-05
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody-05
```


# [06-nbody.cu](06-nbody.cu) - CUDA Streams + change CPU cudaMemPrefetchAsync to GPU updateBodies
```C
__global__ void updateBodies(Body *p, float dt, int n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i<n ; i+=stride)
  {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
  }
}
...
  //MOVE THIS OUT OF LOOP - ONLY NEED TO DO ONCE:
  cudaMemPrefetchAsync(buf, bytes, deviceId);
  checkCuda(cudaDeviceSynchronize());

  for (int iter = 0; iter < nIters; iter++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    //  bodyForce<<<numBlocks, numThreads>>>(p, dt, nBodies); 
    bodyForce<<<numBlocks, numThreads, 0, stream>>>(p, dt, nBodies); 
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    
    // cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId); // TEMPORARY until we move this to GPU !
    // for (int i = 0 ; i < nBodies; i++) { // integrate position
    //   p[i].x += p[i].vx*dt;
    //   p[i].y += p[i].vy*dt;
    //   p[i].z += p[i].vz*dt;
    // }
    updateBodies<<<numBlocks, numThreads, 0, stream>>>(p,dt,nBodies);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    cudaStreamDestroy(stream);

  }
  cudaFree(buf);	
```
```bash
nvcc -std=c++11 -o nbody-06 06-nbody.cu
./nbody-06
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody-06
```

# Profiles
## cuda_gpu_kern_sum
- nbody-02: 5,439,788,609
- nbody-03:    25,924,887
- nbody-04:     4,902,610
- nbody-05:     2,682,757
- nbody-06:     2,685,291 
                  +14,688
## details
### nbody-02
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     66.8   10,173,407,920        513   19,831,204.5   10,145,947.0        7,517  133,326,853  28,213,322.7  poll                  
     32.8    5,001,317,061         10  500,131,706.1  500,137,713.0  500,067,871  500,221,337      49,931.6  pthread_cond_timedwait
      0.3       51,167,034        477      107,268.4        5,296.0          241   25,893,266   1,243,390.1  ioctl                 
      0.0        1,104,352          9      122,705.8      107,148.0        5,686      368,249     103,553.8  sem_timedwait         
      0.0          864,038         27       32,001.4        8,489.0        2,036      503,938      95,122.6  mmap64                
      0.0          188,236         17       11,072.7        2,948.0          912       94,915      23,369.4  mmap                  
      0.0          142,499         44        3,238.6        2,480.0          789       10,139       2,419.2  open64                
      0.0          113,987          3       37,995.7       22,712.0       22,411       68,864      26,733.2  pthread_create        
      0.0           94,410          7       13,487.1        3,982.0        2,406       70,202      25,027.3  munmap                
      0.0           62,614         31        2,019.8        1,385.0          490        8,269       1,892.6  fopen                 
      0.0           23,224         56          414.7           22.0           21       21,870       2,919.2  fgets                 
      0.0           22,816          7        3,259.4        2,276.0           88        7,416       2,979.0  fread                 
      0.0           17,095         11        1,554.1        1,411.0          246        6,711       1,820.4  write                 
      0.0           14,619         25          584.8          513.0          364        2,086         323.4  fclose                
      0.0           14,180         14        1,012.9          777.5          385        1,900         516.3  read                  
      0.0           13,974         60          232.9          140.5           73        2,003         284.7  fcntl                 
      0.0           13,621          6        2,270.2        2,185.0          750        4,454       1,300.5  open                  
      0.0            9,283          3        3,094.3        2,625.0        1,216        5,442       2,151.7  pipe2                 
      0.0            6,986          2        3,493.0        3,493.0        2,293        4,693       1,697.1  socket                
      0.0            5,095          1        5,095.0        5,095.0        5,095        5,095           0.0  connect               
      0.0            1,043          7          149.0          145.0           86          294          71.2  dup                   
      0.0              845          1          845.0          845.0          845          845           0.0  bind                  
      0.0              404          1          404.0          404.0          404          404           0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------
     98.4    5,439,866,724         10  543,986,672.4  543,696,897.5  540,855,415  549,482,415  2,127,172.6  cudaDeviceSynchronize 
      1.6       86,626,614          1   86,626,614.0   86,626,614.0   86,626,614   86,626,614          0.0  cudaMallocManaged     
      0.0          698,828         10       69,882.8       24,250.5       21,861      479,389    143,896.5  cudaLaunchKernel      
      0.0          202,200          1      202,200.0      202,200.0      202,200      202,200          0.0  cudaFree              
      0.0              469          1          469.0          469.0          469          469          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)              Name             
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  -----------------------------
    100.0    5,439,788,609         10  543,978,860.9  543,689,110.5  540,847,504  549,474,714  2,127,238.2  bodyForce(Body *, float, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
     51.6          233,991     36   6,499.8   6,415.5     2,015    11,231      4,502.2  [CUDA memcpy Unified Host-to-Device]
     48.4          219,523     40   5,488.1   5,535.0       991    10,015      4,512.3  [CUDA memcpy Unified Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation              
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
      1.311     40     0.033     0.033     0.004     0.061        0.029  [CUDA memcpy Unified Device-to-Host]
      1.180     36     0.033     0.033     0.004     0.061        0.029  [CUDA memcpy Unified Host-to-Device]
### nbody-03
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)        Name     
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  --------------
     88.0      289,726,587         18  16,095,921.5  10,073,745.0     1,577  128,325,089  31,264,449.4  poll          
     11.5       37,826,714        477      79,301.3       4,763.0       187   15,224,625     787,666.9  ioctl         
      0.3          851,818         27      31,548.8       6,565.0     2,530      459,213      87,849.2  mmap64        
      0.1          290,912          9      32,323.6       6,098.0     3,699      133,774      50,064.2  sem_timedwait 
      0.0          114,381         17       6,728.3       2,206.0     1,188       39,860      10,992.9  mmap          
      0.0           98,250         44       2,233.0       1,878.0       768        5,441       1,174.1  open64        
      0.0           63,015          3      21,005.0      22,184.0    16,135       24,696       4,400.6  pthread_create
      0.0           60,417         31       1,948.9       1,405.0       518        7,857       1,716.9  fopen         
      0.0           30,004          7       4,286.3       2,753.0     1,991       12,671       3,753.8  munmap        
      0.0           20,590         56         367.7          20.0        19       19,383       2,587.2  fgets         
      0.0           18,513          7       2,644.7       2,106.0       129        6,282       1,954.4  fread         
      0.0           14,474         25         579.0         493.0       398        2,398         388.2  fclose        
      0.0           13,265          6       2,210.8       1,913.5       687        4,295       1,368.9  open          
      0.0            9,568         60         159.5          99.0        71        1,639         210.0  fcntl         
      0.0            8,460         14         604.3         483.0       179        1,813         507.4  read          
      0.0            7,546         11         686.0         566.0       221        1,181         365.0  write         
      0.0            6,915          3       2,305.0       1,936.0       923        4,056       1,598.8  pipe2         
      0.0            5,783          2       2,891.5       2,891.5     2,120        3,663       1,091.1  socket        
      0.0            4,750          1       4,750.0       4,750.0     4,750        4,750           0.0  connect       
      0.0            1,037          1       1,037.0       1,037.0     1,037        1,037           0.0  bind          
      0.0              888          7         126.9         110.0        83          197          44.3  dup           
      0.0              438          1         438.0         438.0       438          438           0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------
     71.6       69,117,173          1  69,117,173.0  69,117,173.0  69,117,173  69,117,173          0.0  cudaMallocManaged     
     27.8       26,849,894         10   2,684,989.4   2,622,368.5   2,420,952   2,949,822    182,577.6  cudaDeviceSynchronize 
      0.5          515,370         10      51,537.0       2,761.0       1,583     473,709    148,381.4  cudaLaunchKernel      
      0.1           69,998          1      69,998.0      69,998.0      69,998      69,998          0.0  cudaFree              
      0.0              645          1         645.0         645.0         645         645          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)              Name             
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -----------------------------
    100.0       25,924,887         10  2,592,488.7  2,580,236.0  2,418,029  2,951,273    140,333.5  bodyForce(Body *, float, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
     50.8          234,701     36   6,519.5   6,559.5     2,047    10,944      4,209.0  [CUDA memcpy Unified Host-to-Device]
     49.2          227,468     54   4,212.4   1,088.0     1,024    10,144      4,297.7  [CUDA memcpy Unified Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation              
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
      1.311     54     0.024     0.004     0.004     0.061        0.028  [CUDA memcpy Unified Device-to-Host]
      1.180     36     0.033     0.033     0.004     0.061        0.027  [CUDA memcpy Unified Host-to-Device]

### nbody-04
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)   Min (ns)   Max (ns)    StdDev (ns)        Name     
 --------  ---------------  ---------  ------------  ---------  --------  -----------  ------------  --------------
     81.9      224,098,942         15  14,939,929.5  983,914.0     1,621  139,633,711  35,309,018.5  poll          
     17.3       47,241,298        477      99,038.4    4,642.0       186   14,200,668     963,061.5  ioctl         
      0.4        1,159,992          9     128,888.0  112,213.0    87,321      237,517      46,918.8  sem_timedwait 
      0.3          750,755         27      27,805.7    5,924.0     2,336      450,105      85,184.5  mmap64        
      0.1          164,994         17       9,705.5    2,024.0     1,117       95,305      22,857.7  mmap          
      0.0          105,585         44       2,399.7    1,993.0       784       10,625       1,649.3  open64        
      0.0           57,397         31       1,851.5    1,247.0       452        5,600       1,515.2  fopen         
      0.0           56,780          3      18,926.7   17,470.0    16,568       22,742       3,334.8  pthread_create
      0.0           21,004         56         375.1       20.0        19       19,766       2,638.3  fgets         
      0.0           18,458         14       1,318.4      831.5       371        4,035       1,154.9  read          
      0.0           18,220          6       3,036.7    2,931.5     2,197        4,104         638.7  munmap        
      0.0           17,919          7       2,559.9    1,756.0        79        6,487       2,145.3  fread         
      0.0           13,647          6       2,274.5    1,875.0       713        4,747       1,559.6  open          
      0.0           13,442         25         537.7      485.0       305        1,414         208.9  fclose        
      0.0            8,648         60         144.1      105.0        71          833         113.2  fcntl         
      0.0            7,359          3       2,453.0    2,658.0     1,289        3,412       1,076.2  pipe2         
      0.0            6,981         11         634.6      541.0       261        1,227         372.2  write         
      0.0            5,225          2       2,612.5    2,612.5     2,348        2,877         374.1  socket        
      0.0            3,906          1       3,906.0    3,906.0     3,906        3,906           0.0  connect       
      0.0              862          7         123.1      123.0        80          183          39.6  dup           
      0.0              797          1         797.0      797.0       797          797           0.0  bind          
      0.0              356          1         356.0      356.0       356          356           0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------
     91.7       69,009,874          1  69,009,874.0  69,009,874.0  69,009,874  69,009,874          0.0  cudaMallocManaged     
      7.5        5,639,061         10     563,906.1     496,308.5     409,809     877,736    172,124.1  cudaDeviceSynchronize 
      0.6          486,191         10      48,619.1       1,810.0       1,567     461,322    145,032.5  cudaLaunchKernel      
      0.2          117,809          1     117,809.0     117,809.0     117,809     117,809          0.0  cudaFree              
      0.0              486          1         486.0         486.0         486         486          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)              Name             
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------------------------
    100.0        4,902,610         10  490,261.0  432,878.5   406,879   874,974    141,459.0  bodyForce(Body *, float, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
     54.4          205,423     81   2,536.1   1,088.0     1,054    10,112      3,103.4  [CUDA memcpy Unified Device-to-Host]
     45.6          172,444     13  13,264.9  16,768.0     6,016    17,087      5,001.2  [CUDA memcpy Unified Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation              
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
      1.081     81     0.013     0.004     0.004     0.061        0.020  [CUDA memcpy Unified Device-to-Host]
      0.983     13     0.076     0.098     0.029     0.098        0.031  [CUDA memcpy Unified Host-to-Device]

### nbody-05
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)   Min (ns)   Max (ns)    StdDev (ns)        Name     
 --------  ---------------  ---------  ------------  ---------  --------  -----------  ------------  --------------
     82.3      218,015,404         15  14,534,360.3  964,758.0     2,046  135,066,254  34,168,624.1  poll          
     16.8       44,371,329        497      89,278.3    4,794.0       187   13,739,667     859,477.0  ioctl         
      0.4        1,009,132          9     112,125.8   98,508.0    52,531      240,997      52,982.1  sem_timedwait 
      0.3          829,463         27      30,720.9    4,536.0     1,979      496,685      94,272.9  mmap64        
      0.1          203,307         17      11,959.2    2,691.0     1,134      127,637      30,419.8  mmap          
      0.0           78,117         44       1,775.4    1,574.5       741        3,958         885.9  open64        
      0.0           68,547         31       2,211.2    1,550.0       439       12,187       2,328.2  fopen         
      0.0           66,655          3      22,218.3   19,866.0    18,927       27,862       4,910.1  pthread_create
      0.0           29,206          7       4,172.3    2,008.0        72       18,901       6,568.4  fread         
      0.0           26,069          8       3,258.6    3,270.0     1,950        4,295         740.7  munmap        
      0.0           21,301         56         380.4       21.0        19       20,071       2,679.1  fgets         
      0.0           21,108         14       1,507.7      878.5       393        3,972       1,223.0  read          
      0.0           14,726         25         589.0      500.0       376        2,715         450.2  fclose        
      0.0           12,981          6       2,163.5    1,933.0       724        4,400       1,364.2  open          
      0.0            9,188         60         153.1       89.0        71        2,101         262.6  fcntl         
      0.0            8,236          2       4,118.0    4,118.0     2,731        5,505       1,961.5  socket        
      0.0            7,555          3       2,518.3    2,509.0     1,128        3,918       1,395.0  pipe2         
      0.0            7,244         11         658.5      635.0       267        1,252         357.8  write         
      0.0            4,913          1       4,913.0    4,913.0     4,913        4,913           0.0  connect       
      0.0            1,147          1       1,147.0    1,147.0     1,147        1,147           0.0  bind          
      0.0              866          7         123.7      136.0        74          186          43.2  dup           
      0.0              389          1         389.0      389.0       389          389           0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------
     94.4       67,973,278          1  67,973,278.0  67,973,278.0  67,973,278  67,973,278          0.0  cudaMallocManaged     
      4.0        2,908,949         20     145,447.5     145,521.5       5,279     271,330    128,473.1  cudaDeviceSynchronize 
      0.7          484,588         20      24,229.4      26,858.5       6,428     118,557     25,005.9  cudaMemPrefetchAsync  
      0.7          472,177         10      47,217.7       1,968.5       1,487     451,700    142,124.3  cudaLaunchKernel      
      0.2          150,745          1     150,745.0     150,745.0     150,745     150,745          0.0  cudaFree              
      0.0              502          1         502.0         502.0         502         502          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)              Name             
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------------------------
    100.0        2,682,757         10  268,275.7  268,400.5   267,888   268,625        248.0  bodyForce(Body *, float, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
     52.8          169,621     10  16,962.1  16,975.0    16,607    17,150        147.9  [CUDA memcpy Unified Host-to-Device]
     47.2          151,672     10  15,167.2  15,167.0    15,135    15,200         21.3  [CUDA memcpy Unified Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation              
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
      0.983     10     0.098     0.098     0.098     0.098        0.000  [CUDA memcpy Unified Device-to-Host]
      0.983     10     0.098     0.098     0.098     0.098        0.000  [CUDA memcpy Unified Host-to-Device]
### nbody-06
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)        Name     
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  --------------
     77.9      243,669,115         15  16,244,607.7  1,111,490.0     1,533  159,458,242  40,335,796.6  poll          
     21.4       66,923,198        498     134,383.9      4,484.5       187   33,129,907   1,647,079.5  ioctl         
      0.3          939,510         27      34,796.7      6,809.0     2,308      458,043      91,362.2  mmap64        
      0.2          720,015          9      80,001.7     73,284.0     6,979      191,242      56,698.8  sem_timedwait 
      0.1          162,397         17       9,552.8      2,897.0     1,001       85,395      20,538.7  mmap          
      0.0          102,614         44       2,332.1      2,019.0       726        4,829       1,127.5  open64        
      0.0           77,150          3      25,716.7     26,975.0    17,937       32,238       7,233.1  pthread_create
      0.0           67,599         31       2,180.6      1,554.0       479       11,366       2,246.6  fopen         
      0.0           25,582          7       3,654.6      2,084.0        62        9,490       3,339.7  fread         
      0.0           23,394         56         417.8         21.0        20       22,100       2,950.1  fgets         
      0.0           20,440          7       2,920.0      3,036.0     1,869        4,073         842.8  munmap        
      0.0           15,150         25         606.0        499.0       320        2,656         441.6  fclose        
      0.0           14,685          6       2,447.5      2,404.0       716        4,230       1,417.2  open          
      0.0           10,700         14         764.3        806.5       190        1,459         375.6  read          
      0.0            9,989         60         166.5        100.0        70        2,050         258.8  fcntl         
      0.0            7,710          3       2,570.0      2,667.0     1,006        4,037       1,517.8  pipe2         
      0.0            7,198         11         654.4        591.0       222        1,503         411.2  write         
      0.0            7,060          2       3,530.0      3,530.0     2,510        4,550       1,442.5  socket        
      0.0            4,525          1       4,525.0      4,525.0     4,525        4,525           0.0  connect       
      0.0              866          7         123.7        127.0        74          187          47.9  dup           
      0.0              742          1         742.0        742.0       742          742           0.0  bind          
      0.0              300          1         300.0        300.0       300          300           0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------
     95.2       71,159,534          1  71,159,534.0  71,159,534.0  71,159,534  71,159,534          0.0  cudaMallocManaged     
      3.7        2,751,199         21     131,009.5       7,254.0       2,485     271,573    136,468.0  cudaDeviceSynchronize 
      0.8          567,129         20      28,356.5       2,450.5       1,881     462,241    102,596.1  cudaLaunchKernel      
      0.2          126,369          1     126,369.0     126,369.0     126,369     126,369          0.0  cudaMemPrefetchAsync  
      0.2          113,249          1     113,249.0     113,249.0     113,249     113,249          0.0  cudaFree              
      0.1           38,877         10       3,887.7       1,261.0       1,073      21,796      6,398.1  cudaStreamCreate      
      0.0           21,204         10       2,120.4       1,253.0       1,125       7,468      1,982.1  cudaStreamDestroy     
      0.0              619          1         619.0         619.0         619         619          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                Name              
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  --------------------------------
     99.5        2,685,291         10  268,529.1  268,449.0   267,793   269,522        506.4  bodyForce(Body *, float, int)   
      0.5           14,688         10    1,468.8    1,472.0     1,408     1,504         28.0  updateBodies(Body *, float, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
    100.0           16,639      1  16,639.0  16,639.0    16,639    16,639          0.0  [CUDA memcpy Unified Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation              
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
      0.098      1     0.098     0.098     0.098     0.098        0.000  [CUDA memcpy Unified Host-to-Device]