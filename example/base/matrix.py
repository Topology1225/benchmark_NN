# 科学計算用のモジュール
import numpy as np


def main():
    # 行列の計算
    ## 行列Aを作成
    A = np.array([[1,2], [3,1]]) 
    # print(A)

    # 行列Bを作成
    B = np.array([[2,5],[6,1]])
    # print(B)

    ## 行列積の計算 
    AB = np.matmul(A,B)
    print(AB)

    ## 要素積の計算

    # 微積の計算
    ## 指数関数
    a = 1
    exp_a = np.exp(a)
    print(exp_a)

if __name__=="__main__":
    main()