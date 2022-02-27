## MMOE复现

* 数据描述：

  * feature len11，label len2-点击/转化

    ```python
    ['user_id', 'item_id', 'sdk_type', 'remote_host', 'device_type', 'dtu', 'click_goods_num', 'buy_click_num', 'goods_show_num','goods_click_num', 'brand_name']
    ```

  * 17时9365条数据，18时20959条数据

  * 正负样本比例统计：

    17时 label 1-0: 77/ label 1-1: 28/ label 0-0: 9232

    18时 label 1-0: 196/ label 1-1: 96/ label 0-0: 20571

* 实验结果：TRAIN-17时数据，TEST-18时数据；负样本全采样

|      | auc:ctr           | auc:ctcvr         | auc:cvr | config                                       |
| ---- | ----------------- | ----------------- | ------- | -------------------------------------------- |
| DNN  | 0.992844/0.971075 | 0.959630/0.959630 |         | embed8,batch32,epoch30000,mlp[64,32]         |
| DFM  | 0.990795/0.973119 | 0.931434/0.871075 |         | embed8,batch32,epoch30000,mlp[64,32]         |
| DCN  | 0.993066/0.974023 | 0.973211/0.961370 |         | embed8,batch32,epoch30000,mlp[64,32],cln2    |
| MMOE | 0.981141/0.978643 | 0.989741/0.969144 |         | embed8,batch32,epoch30000,mlp[32,16],expert3 |
| ESMM | 0.987891/0.983463 |                   |         | embed8,batch32,epoch30000,mlp[32,16],expert1 |



