
执行readAsNumber.cs可以将心电图数据为长度40320的base64的字符串，如：

```
vwe+B7wHuwe6B7oHuQe4B7gHtwe1B7QHswexB7AHrgetB6wHq
```

转换为byte类型数组，每两位拼成一个int32的数字，低位在前，高位在后。如

```
1983 1982 1980 1979 …………2056 2058 2059
```

执行**readTxtAndsaveImage.py**可以将txt文件中保存的上述int32数字用python的**matp
lotlib.pyplot**画出并保存下来。