#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
	using namespace std::chrono;

	int N = 1000;
	if (argc == 2) {
		N = atoi(argv[1]);
	}
	printf("Using %d x %d\n", N, N); 

	float *M  = new float[N*N];
	float *K  = new float[N];
	float *g1 = new float[2*N];
	float k = 1.3;

	for (int i=0; i<N; i++)
		K[i]  = rand() / (float)RAND_MAX;
	for (int i=0; i<2*N; i++)
		g1[i] = rand() / (float)RAND_MAX;

	auto t0 = high_resolution_clock::now();
	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
			M[i+j*N] = g1[N+i-j] * (K[i]+k) * (K[j]+k);
	duration<float> t = high_resolution_clock::now() - t0;
	printf("CPU took %f seconds\n", t.count());
	
	return 0;
}
