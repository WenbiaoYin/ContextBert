# ContextBert

Code for IJCNN2022 paper: [“ContextBert: Enhanced Implicit Sentiment Analysis Using Implicit-sentiment-query Attention”](https://ieeexplore.ieee.org/abstract/document/9892878)

## Overview

In sentiment analysis, it is concerned recently that a significant portion of subjective sentences across different domains do not contain explicit sentiment words but still convey clear subjective sentiment, which is known as implicit sentiment. However, currently available methods do not perform well on implicit sentiment analysis. In most cases implicit sentiment can be inferred from the context. To address this issue, we propose the ContextBert model, which employs implicit-sentiment-query attention to find the contextual information related to the target sentence to enhance implicit sentiment analysis. Experimental results show that our method achieves state-of-the-art performance on SMP2019-ECISA and EmoContext-Implicit (reconstructed based on EmoContext, to enrich the implicit sentiment analysis dataset).

![model.png](https://github.com/WenbiaoYin/ContextBert/blob/master/model.png?raw=true)

## Dataset

[SMP - ECISA 2019](https://www.biendata.xyz/competition/smpecisa2019/ )

[EmoContex](https://aclanthology.org/S19-2005/)



## Citation

If you found this repository useful, please [cite](https://scholar.googleusercontent.com/scholar.bib?q=info:c1LcpEcbWXwJ:scholar.google.com/&output=citation&scisdr=CgUsLB-yEIboqcdUAhI:AAGBfm0AAAAAY1JSGhLs0r8QZEbvnDJdluQsnRk7SyZk&scisig=AAGBfm0AAAAAY1JSGj2bcJxSw0X-oGNVCff9w3i2VMw-&scisf=4&ct=citation&cd=-1&hl=zh-CN) our paper:

```
@inproceedings{yin2022contextbert,
  title={ContextBert: Enhanced Implicit Sentiment Analysis Using Implicit-sentiment-query Attention},
  author={Yin, Wenbiao and Shang, Lin},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```

