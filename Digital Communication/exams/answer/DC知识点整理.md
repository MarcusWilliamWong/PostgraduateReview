# DC知识点整理

## 调制与解调

![16QAM格雷码](D:\Learning Notes for CS\EmbeddedProgramming\assets\16QAM格雷码.png)

## 波特率与比特率

符号率:符号率也称波特率，是衡量每秒在通信信道上传输的符号数的度量。一个符号表示一个特定的值或状态，可以是一个比特或一组比特，具体取决于所使用的调制方案。在数字通信系统中，符号用于表示信息，不同的调制方案 (如正交振幅调制或QAM) 可以用于将多个比特编码成一个符号。符号率通常用波特 (baud) 或每秒传输的符号数 (sps) 来衡量
比特率: 比特率也称数据率或比特速率，是衡量每秒在通信信道上传输的比特数的度量。它代表了通信系统传递数字信息的总体能力，直接影响在给定时间内可以传输的数据量。比特率受到调制方案、符号率和每个符号编码的比特数等因素的影响。比特率通常用比特每秒(bps) 来衡量

## 信号带宽

一个信号所包含谐波的最高频率与最低频率之差，即该信号所拥有的频率范围，定义为该信号的带宽。一个由数个正弦波叠加成的方波信号，其最低频率分量是其基频，假定为f=2kHz，其最高频率分量是其7次谐波频率，即7f =7×2=14kHz，因此该信号带宽为7f - f =14-2=12kHz。

## 升余弦滚降滤波器

$\frac{1}{T_{sym}} = \frac{B_{DSB}}{1+\beta}$

$T_{sym}$表示符号间隔时间，symbol period

$B_{DSB}$表示带宽，bandwidth，BW

$\beta$表示滚降系数，roll-off一般为0.5



## 格雷码（Gray code）

手动构造格雷码：

k位的格雷码可以通过以下方法构造。我们从全格雷码开始，按照下面策略：1.翻转最低位得到下一个格雷码，如000 -> 001；2.把最右边的1的左边的位翻转得到下一个格雷码，例如001->011；

交替按照上述策略生成2^k-1^次，可得到k位的格雷码序列。

分治法：从n-1位格雷码到n位格雷码：原来码的基础上最高位补零，再最高位都补1，这些补1的格雷码剩余位是原来n-1位格雷码镜像翻转下来

https://blog.csdn.net/qq_50737715/article/details/123587206

格雷码的好处：



## 信息（information）与信息熵（source entropy）

$I_A = \sum^{N-1}_{i=0}\log\frac{1}{P_A}$ 针对的是一串特定数据

$H=\sum^{N-1}_{i=0}P_i\log_2\frac{1}{P_i}$ 求一个可能字符集合（set）的平均信息量



## 霍夫曼编码（Huffman code）

https://www.youtube.com/watch?v=dM6us854Jk0

## LZW 压缩编码

假定两个指针，P和C，P指向上一个字符，从空开始；C指向当前字符，从传输字符串第一个字符开始，依次往下指；初始字符查找表应包含传输字符串中所有的单字符编码。

规则：初始化PC后，每次都查看合并后的PC是否在表中，^1^如果PC没有在表中，则添加编码表，并对P指向的字符编码传输字符串，然后让P=C，C往下指一个字符；如果PC在表中，则令P=PC，C往下指一个字符，注意不需要对传输字符串编码；重复以上过程，从^1^处开始

## Discrete Memoryless Source

离散无记忆源：一个符号的出现概率与之前出现的符号无关

## 加性高斯白噪声（Additive White Gaussian Noise, AWGN）

这个公式也就是Shannon-Hartley theorem。任何通信信道内，能够准确无误地传输数据的最大速度与噪声和带宽有关。他将这个最大比特率称为“信道容量”，也就是目前众所周知的“香农极限”

$C = B\log{(1+\frac{S}{N})}$

C: channel capacity(bps)

B: bandwidth (Hz)

S/N: signal-to-noise ratio, where S is the signal power and N = N~0~B

S/N就是SNR，单位就是dB；S 表示接收的信号平均功率 (W)，N 表示平均噪声功率 (W)

## 信道编码

差错控制方法：前向纠错法，所发码带有一定的纠错能力，不需要反向信道；检错重发法（ Automatic Resend Request, ARQ）；混合纠错法（HEC），在纠错范围内，纠正，超出纠错范围，ARQ重发

### 前向纠错（Forward Error Correction）

Forward error correction (FEC) or channel coding is a technique used for
controlling errors in data transmission over unreliable or noisy communication
channels. The central idea is the sender encodes the message in a redundant way by
using an error-correcting code.

### ASCII码与UTF-8编码

ASCII最开始有128个，用7位，一般第8位用奇偶校验作为校验码，后来扩展到256个，使用8位编码；UTF-8是一种可变长编码规则，每8位作为一个编码单元，前8位依旧使用ASCII编码，所以ASCII是UTF-8的子集

UTF-8 代表 8 位一组表示 Unicode 字符的格式，使用 1 - 4 个字节来表示字符

### 汉明距离（Hamming distance）

A code with all distinct code words has a minimum Hamming distance of at
least 1
 A code which permits detection of up to e errors per word has a minimum
Hamming distance of at least (e+1)
 A code which permits correction of up to e errors per word has a minimum
Hamming distance of at least (2e+1)

例如：带有r位汉明码的纠错能力应该是$e = \frac{d_{min}-1}{2}$，且汉明码的最小汉明距离一定为3，所以汉明码纠错能力为1。

### 奇偶校验码（Odd/Even Parity Code）

奇/偶校验，校验码补0/1保证传输数据中的1是奇/偶数个

### 汉明码（Hamming Code）

total length/block length $n=2^r -1$ 总码长n，校验位数r

data bits/message length $k = 2^r - 1 - r$ 信息位k

code-rate efficiency $\frac{k}{n}=1-\frac{r}{2^r-1}$，当r越大，即n越大时，编码效率就越好，越接近于1

hamming distance_min = 3

汉明码纠错能力为t=1

Hamming(n, k)

![hammingcode](D:\Learning Notes for CS\EmbeddedProgramming\assets\hammingcode.png)

![hamming(7,4)](D:\Learning Notes for CS\EmbeddedProgramming\assets\hamming(7,4).png)

生成矩阵G：信息位为单位对角矩阵，求校验位的矩阵，信息位矩阵与校验位矩阵合并起来就是生成矩阵G

校验矩阵H：让r位校验位成单位对角矩阵，且所有影响对应校验位的信息位上应该都填1，得到的信息位矩阵，信息位矩阵和校验位单位对角矩阵合起来就是校验矩阵H。

可以对照上边的圆圈图看校验位和信息位的对应影响关系

$G=[I_k|P]$

$H=[P^T|I_r]$

使用纠错矩阵纠错：![纠错矩阵的使用](D:\Learning Notes for CS\EmbeddedProgramming\assets\纠错矩阵的使用.png)

![hamming(7,4)GH](D:\Learning Notes for CS\EmbeddedProgramming\assets\hamming(7,4)GH.png)

### 低密度奇偶校验代码（Low-density parity-check code, LDPC）

也就是Gallager码，

LDPC codes are capacity-approaching codes, which means that practical

constructions exist that allow the noise threshold to be set very close to the

Shannon limit

### 交错数据（Interleaving）

一般错误都是burst error，就是集中数据的某段爆发发生错误，这样可能会导致错误数量大于数据块校验码的纠错能力，导致无法正常获取正确数据；采用交错这样的手段，先在传输前交错数据，再传输（这里发生错误），然后再逆交错接收到的数据，可以把爆发性错误分散到不同数据块中，再解码纠错完成传输。

### 循环码（Cyclic Code）与码多项式

用代数多项式表示循环码字，这种多项式叫码多项式。除了全零码外，幂次最低的码多项式称为生成多项式

循环码(7, 3)指的是码长7位，3个位为0，其余为1.

![码多项式](D:\Learning Notes for CS\EmbeddedProgramming\assets\码多项式.png)

![循环码的生成多项式](D:\Learning Notes for CS\EmbeddedProgramming\assets\循环码的生成多项式-1683305073951-5.png)

![寻找循环码的生成多项式](D:\Learning Notes for CS\EmbeddedProgramming\assets\寻找循环码的生成多项式-1683305076924-7.png)

### 循环冗余校验码（Cyclic Redundancy Check, CRC）

使用生成多项式与二进制数据补零r位（r是生成多项式的最高次幂）后的数据做按位异或操作，即模二除法，获得的三位余数即为校验码；接收端，获取n+r位数据，继续使用相同的生成多项式与接收数据做按位异或，如果余数是r位0，则无错误；如果不均为0，则错误位为余数所示的位置，注意数据右边是低位。

From Hamming distance concept, a code with minimum distance 2t+1 can correct any t errors.A linear block code that corrects all burst errors of length t or less must have at least 2t check symbols. 根据最小汉明距离和纠错能力位数的关系可知，长度为t的码，需要使用长度为2t的线性编码校验位

### BCH code（Bose–Chaudhuri–Hocquenghem）

BCH码是循环码的子集

### 卷积码（Binary Convolutional Code）
