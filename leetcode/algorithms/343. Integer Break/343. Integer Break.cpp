#include<iostream>
using namespace std;

class Solution {
public:
	int integerBreak(int n) {
		if (n == 2) return 1;
		if (n == 3) return 2;
		int res = 1;
		while (n > 4)
		{
			res *= 3;
			n = n - 3;
		}
		return res*n;
	}
};

int main()
{
	Solution s;
	int result = s.integerBreak(13);
	system("pause");
	return 0;
}