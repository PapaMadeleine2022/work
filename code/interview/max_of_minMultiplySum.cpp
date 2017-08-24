#include<iostream>
using namespace std;

class Solution {
public:
	int max_of_minMultiplySum(int arr[], int size) {
		int sum=0, min=0;
		int resMax = INT_MIN;
		for (int i = 0; i < size; i++)
		{
			int j=i, k = i;
			while (k<size)
			{
				min = arr[j];
				for (int m = j; m < k + 1; m++)
				{
					if (arr[m] < min)min = arr[m];
					sum += arr[m];
				}
				if (min*sum>resMax) resMax = min*sum;
				sum = 0;
				k++;
			}
		}
		return resMax;
	}
};
int main()
{
	Solution s;
	int arr[] = { 6, 2, 1 };
	return s.max_of_minMultiplySum(arr, sizeof(arr) / sizeof(int));
}