
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <clocale>


cudaError_t mulWithCuda(int *c, const int *a, const int *b, unsigned int size);

// Точка входа в GPU
__global__ void mulKernel(int *c, const int *a, const int *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * b[i];
	//printf("{1,2,3,4,5} * {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", //можно вывести на экран средствами GPU
	//	c[0], c[1], c[2], c[3], c[4]);
}

// Точка входа в приложение
int main()
{
	setlocale(LC_CTYPE, "rus");
	
	const int arraySize = 8000;
	int a[arraySize] = { 0 };
	int b[arraySize] = { 0 };
	int c[arraySize] = { 0 };

	for (int i = 0; i < arraySize; i++)
	{
		a[i] = i + 1;
		b[i] = a[i] * 10;
	}

	srand(time(0));

	// Add vectors in parallel.
	cudaError_t cudaStatus = mulWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mulWithCuda failed!");
		return 1;
	}

	printf("Время работы метода: %d мс \n", clock());


	printf("\n*******************************************\n");
	printf("Вывод первых и последних 10 результатов перемножения массивов размерностью 8000 элементов: \n\n");
	printf("{1,2,3,4,5,6,7,8,9,10} * {10,20,30,40,50,60,70,80,90,100} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
	printf("{7991,7992,7993,7994,7995,7996,7997,7998,7999,8000} * {79910,79920,79930,79940,79950,79960,79970,79980,79990,80000} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
		c[7990], c[7991], c[7992], c[7993], c[7994], c[7995], c[7996], c[7997], c[7998], c[7999]);

	printf("\n*******************************************\n\n");

	printf("Основные данные по устройству: \n\n");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int device = 0; device < deviceCount; device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Номер устройства: %d\n", device);
		printf("Имя устройства: %s\n", deviceProp.name);
		printf("Объем глобальной памяти: %d\n", deviceProp.totalGlobalMem);
		printf("Объем shared-памяти в блоке : %d\n", deviceProp.sharedMemPerBlock);
		printf("Объем регистровой памяти: %d\n", deviceProp.regsPerBlock);
		printf("Размер warp'a: %d\n", deviceProp.warpSize);
		printf("Размер шага памяти: %d\n", deviceProp.memPitch);
		printf("Макс количество потоков в блоке: %d\n", deviceProp.maxThreadsPerBlock);

		printf("Максимальная размерность потока: x = %d, y = %d, z = %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);

		printf("Максимальный размер сетки: x = %d, y = %d, z = %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);

		printf("Тактовая частота: %d\n", deviceProp.clockRate);
		printf("Общий объем константной памяти: %d\n", deviceProp.totalConstMem);
		printf("Вычислительная мощность: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Величина текстурного выравнивания : %d\n", deviceProp.textureAlignment);
		printf("Количество процессоров: %d\n", deviceProp.multiProcessorCount);
	}
	printf("\n*******************************************\n\n");

	printf("Расчёт на хосте\n");
	srand(time(0));

	for (int i = 0; i < arraySize; i++)
	{
		c[i] = a[i] * b[i];
	}
	printf("Время работы метода на хосте: %d мс \n", clock());
	printf("Вывод первых и последних 10 результатов перемножения массивов размерностью 8000 элементов: \n\n");
	printf("{1,2,3,4,5,6,7,8,9,10} * {10,20,30,40,50,60,70,80,90,100} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
	printf("{7991,7992,7993,7994,7995,7996,7997,7998,7999,8000} * {79910,79920,79930,79940,79950,79960,79970,79980,79990,80000} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
		c[7990], c[7991], c[7992], c[7993], c[7994], c[7995], c[7996], c[7997], c[7998], c[7999]);

	printf("\n*******************************************\n\n");

			   
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mulWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0; // dev - находится на GPU
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0); // указываем, что работаем на "0"-й карте, м.б. несколько
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); // выделяем память на переменную
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); // выделяем память на переменную
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int)); // выделяем память на переменную
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); // копируем значения переменной с хоста на GPU 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice); // копируем значения переменной с хоста на GPU 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	
	// Launch a kernel on the GPU with one thread for each element.
	dim3 block(64, 1);
	dim3 grid((size / 64), 1);
	mulKernel << <grid, block >> > (dev_c, dev_a, dev_b); // запуск функции с параметрами (size - размер массива)
	   

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mulKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
