using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;


namespace ConsoleApplication1
{
    class Write
    {
        public void write(string path, string res_str)
        {
            FileStream fs = new FileStream(path, FileMode.Create);
            StreamWriter sw = new StreamWriter(fs);
            //开始写入
            sw.Write(res_str);
            //清空缓冲区
            sw.Flush();
            //关闭流
            sw.Close();
            fs.Close();
        }
    }
    class TraversalDirectory
    {
        public string[] getDirectory(string folderFullName)
        {
            DirectoryInfo TheFolder = new DirectoryInfo(folderFullName);

            DirectoryInfo[] dirInfo = TheFolder.GetDirectories();

            //遍历文件夹
            String[] res = new String[dirInfo.Length];
            int index=0;
            foreach (DirectoryInfo NextFolder in dirInfo)
            {
                Console.WriteLine(NextFolder.FullName);
                res[index] = NextFolder.FullName;
                index++;
            }
            return res;
        }
        public string[] getFile(string folderFullName)
        {
            DirectoryInfo TheFolder = new DirectoryInfo(folderFullName);

            FileInfo[] fileInfo = TheFolder.GetFiles();
            //遍历文件
            String[] res = new String[fileInfo.Length];
            int index = 0;
            foreach (FileInfo NextFile in fileInfo)
            {
                Console.WriteLine(NextFile.FullName);
                res[index] = NextFile.FullName;
                index++;
            }
            return res;
        }
        
    }
    class Read
    {
        public string read(string path)
        {
            StreamReader sr = new StreamReader(path, Encoding.Default);
            String s="";
            String line;
            while ((line = sr.ReadLine()) != null)
            {
                s = s + line;
            }
            //关闭流
            sr.Close();
            return s;
        }

    }
    class ReadAndWrite
    {
        public void readAndwrite(string filePath)
        {
            Read r = new Read();
            String s = r.read(filePath);
            //bool isEmpty = (s == "");
            if (s == "")
            {
                Console.WriteLine(filePath);
            }
            byte[] b = Convert.FromBase64String(s);
            //数据转化：
            String res_str = "";
            int[] res = new int[b.Length / 2 + 1];
            //int j = 0;
            for (int i = 0; i < b.Length; i += 2)
            {
                int convert = (b[i] & 0xFF) | (b[i + 1] << 8) & 0xFF00;
                res_str = res_str + convert.ToString() + " ";
            }
            Write w = new Write();
            String path = filePath + ".txt";
            w.write(path, res_str);
        }
    }
    class Program
    {

        static void Main(string[] args)
        {
            TraversalDirectory td = new TraversalDirectory();
            String[] dirs = td.getDirectory("C:\\Users\\Seven\\Desktop\\样本数据_20170308\\样本数据");
            for (int i = 0; i < dirs.Length; ++i)
            {
                String[] files = td.getFile(dirs[i]);
                ReadAndWrite rw = new ReadAndWrite();
                for (int j = 0; j < files.Length; ++j)
                {
                    rw.readAndwrite(files[j]);
                }
            }
        }
    }

}
