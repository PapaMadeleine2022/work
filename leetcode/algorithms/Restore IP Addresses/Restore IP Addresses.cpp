#include<vector>
#include<iostream>
using namespace std;

//using backtracking
class Solution {
public:
	vector<string> result;
	vector<string> restoreIpAddresses(string s) {
		backTracing(s, 0, "");
		return  result;
	}
	int strToint(string s)
	{
		if (s == "") return -1;
		int res = 0;
		for (int i = 0; i < s.length(); ++i)
		{
			int j = s[i] - '0';
			res = res * 10 + j;
		}
		return res;
	}
	//s represents the left string, sec represents the sec th ip section(start from 0), curIpAddress represents the restored ip
	void backTracing(string left, int sec, string curIpAddress){
		if (left.length() < 4 - sec || left.length() > 3 * (4 - sec))return;
		if (sec == 3)
		{
			if (left.length() >1 && left[0] == '0') return;
			int leftToInt = strToint(left);
			if (leftToInt >= 0 && leftToInt <= 255)
			{
				curIpAddress += left;
				result.push_back(curIpAddress);
			}
		}
		for (int i = 1; i < 4 && i < left.length(); ++i)
		{	
			string tmp = left.substr(0, i);
			if (tmp.length() >1 && tmp[0] == '0') return;
			int tmpToInt = strToint(tmp);
			if (tmpToInt >= 0 && tmpToInt <= 255)
			{
				//notice: you can not use "curIpAddress=curIpAddress + tmp + '.'" here
				string nextIpAddress = curIpAddress + tmp + '.';
				backTracing(left.substr(i), sec + 1, nextIpAddress);
			}
		}
	}
};

int main()
{
	Solution s;
	vector<string> result = s.restoreIpAddresses("12345");
	system("pause");
	return 0;
}