## 装饰器注解

### 注解不带参数

```python
@dec2
@dec1
def func(arg1, arg2, ...):
	pass
```

这等价于

```python
def func(arg1, arg2, ...):
    pass
func = dec2(dec1(func))
```

### 注解带参数

```python
@decomaker(argA, argB, ...)
def func(arg1, arg2, ...):
```

这等价于

```python
func = decomaker(argA, argB, ...)(func)
```



### 一个例子

```python
def spamrun(fn):
    def sayspam(*args):
        print("spam, spam")
        fn(*args)
    return saysapm

@spamrun
def useful(a,b):
    print(a*b)
    
if __name__=="__main__"
    useful(2, 5)
```

这等价于

```python
def spamrun(fn)
	...
def useful(a,b)
	...
if __name__=="__main__"
    useful=spamrun(useful)
    useful(2,5)
#或者
if __name__=="__main__"
    useful=spamrun(useful)(2,5)
```

spamrun(useful)返回了一个函数sayspam，输入则是（2，5），在sayspam中输出了spam, 同时调用了useful。最终输出为：

```python
spam, spam
10
```

### 参考

[装饰器](https://zhuanlan.zhihu.com/p/45458873)

[functools.wraps深入理解](https://zhuanlan.zhihu.com/p/45535784)



## 协变(Covariant)、逆变(Contravariant)、不变(Invariant)

### What is?

如果 A<:B， 则可以在任何地方用A替换B

> 例如：子类A继承父类B，则可以在任何地方用A替换B

对于某类变换C

协变: 

```
A<:B => C[A]<: C[B]
```

逆变

```
A<:B => C[B] >: C[A]
```

不变：

C[A]与C[B]不能互相交换

### 一个例子[todo]

```python
import abc

from typing import Generic, TypeVar

class Base:
    def foo(self):
        print("foo")

class Derived(Base):
    def bar(self):
        print("bar")
```

子类Derived <: 父类Base





