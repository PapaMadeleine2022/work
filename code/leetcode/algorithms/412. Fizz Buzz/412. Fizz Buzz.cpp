#include<vector>
#include<iostream>
#include <string>
using namespace std;

class Solution {
public:
	vector<string> fizzBuzz(int n) {
		vector<string> res;
		char arr[100];
		for (int i=1; i <=n; ++i)
		{
			if (i % 3 == 0 && i % 5 == 0)
			{
				res.push_back("FizzBuzz");
				continue;
			}
			else if (i % 3 == 0)
			{
				res.push_back("Fizz");
				continue;
			}
			
			else if (i % 5 == 0)
			{
				res.push_back("Buzz");
				continue;
			}
			else
				res.push_back(to_string(i));
		}
		return res;
	}
};
int main()
{
	Solution s;
	vector<string> result = s.fizzBuzz(15);
	system("pause");
	return 0;
}