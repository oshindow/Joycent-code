# KeSpeech province mapping

Source metadata:

- `/data1/xintong/datasets/kespeech/KeSpeech/Metadata/spk2city`
- `/data1/xintong/datasets/kespeech/KeSpeech/Metadata/city2Chinese`
- `/data1/xintong/datasets/kespeech/KeSpeech/Metadata/city2subdialect`
- `/data1/xintong/datasets/kespeech/KeSpeech/Metadata/subdialect2Chinese`

`city_province_subdialect.txt` is a headerless, pipe-separated `city|province_or_municipality|regional_subdialect` mapping.

`spk2province.txt` and `spk2regional_subdialect.txt` are derived by joining the official `spk2city` with the city mapping.

Important: `regional_subdialect` is the city-level dialect-region prior from official `city2subdialect`; it is not the utterance-level label in `phase1.utt2subdialect`, which may be `Mandarin` or a regional subdialect for different utterances from the same speaker.

## City mapping

| City | City (Chinese) | Province/municipality | Chinese | Regional subdialect | Chinese |
|---|---|---|---|---|---|
| Hefei | 合肥市 | Anhui | 安徽省 | Jiang-Huai | 江淮官话 |
| Beijing | 北京市 | Beijing | 北京市 | Beijing | 北京官话 |
| Chongqing | 重庆市 | Chongqing | 重庆市 | Southwestern | 西南官话 |
| Lanzhou | 兰州市 | Gansu | 甘肃省 | Lan-Yin | 兰银官话 |
| Guilin | 桂林市 | Guangxi | 广西壮族自治区 | Southwestern | 西南官话 |
| Guiyang | 贵阳市 | Guizhou | 贵州省 | Southwestern | 西南官话 |
| Baoding | 保定市 | Hebei | 河北省 | Ji-Lu | 冀鲁官话 |
| Chengde | 承德市 | Hebei | 河北省 | Beijing | 北京官话 |
| Langfang | 廊坊市 | Hebei | 河北省 | Beijing | 北京官话 |
| Shijiazhuang | 石家庄市 | Hebei | 河北省 | Ji-Lu | 冀鲁官话 |
| Tangshan | 唐山市 | Hebei | 河北省 | Ji-Lu | 冀鲁官话 |
| Xingtai | 邢台市 | Hebei | 河北省 | Ji-Lu | 冀鲁官话 |
| Harbin | 哈尔滨市 | Heilongjiang | 黑龙江省 | Northeastern | 东北官话 |
| Zhengzhou | 郑州市 | Henan | 河南省 | Zhongyuan | 中原官话 |
| Chifeng | 赤峰市 | InnerMongolia | 内蒙古自治区 | Beijing | 北京官话 |
| Nanjing | 南京市 | Jiangsu | 江苏省 | Jiang-Huai | 江淮官话 |
| Yangzhou | 扬州市 | Jiangsu | 江苏省 | Jiang-Huai | 江淮官话 |
| Changchun | 长春市 | Jilin | 吉林省 | Northeastern | 东北官话 |
| Chaoyang | 朝阳市 | Liaoning | 辽宁省 | Beijing | 北京官话 |
| Dalian | 大连市 | Liaoning | 辽宁省 | Jiao-Liao | 胶辽官话 |
| Dandong | 丹东市 | Liaoning | 辽宁省 | Jiao-Liao | 胶辽官话 |
| Shenyang | 沈阳市 | Liaoning | 辽宁省 | Northeastern | 东北官话 |
| Yingkou | 营口市 | Liaoning | 辽宁省 | Jiao-Liao | 胶辽官话 |
| Shizuishan | 石嘴山市 | Ningxia | 宁夏回族自治区 | Lan-Yin | 兰银官话 |
| Wuzhong | 吴忠市 | Ningxia | 宁夏回族自治区 | Lan-Yin | 兰银官话 |
| Yinchuan | 银川市 | Ningxia | 宁夏回族自治区 | Lan-Yin | 兰银官话 |
| Zhongwei | 中卫市 | Ningxia | 宁夏回族自治区 | Lan-Yin | 兰银官话 |
| XiAn | 西安市 | Shaanxi | 陕西省 | Zhongyuan | 中原官话 |
| Jinan | 济南市 | Shandong | 山东省 | Ji-Lu | 冀鲁官话 |
| Qingdao | 青岛市 | Shandong | 山东省 | Jiao-Liao | 胶辽官话 |
| Weihai | 威海市 | Shandong | 山东省 | Jiao-Liao | 胶辽官话 |
| Yantai | 烟台市 | Shandong | 山东省 | Jiao-Liao | 胶辽官话 |
| Chengdu | 成都市 | Sichuan | 四川省 | Southwestern | 西南官话 |
| Kunming | 昆明市 | Yunnan | 云南省 | Southwestern | 西南官话 |

## Speakers per province/municipality

| Province/municipality | Chinese | Speakers |
|---|---|---:|
| Hebei | 河北省 | 5545 |
| Liaoning | 辽宁省 | 3266 |
| Shandong | 山东省 | 2457 |
| Heilongjiang | 黑龙江省 | 2215 |
| Shaanxi | 陕西省 | 2003 |
| Anhui | 安徽省 | 1954 |
| Jiangsu | 江苏省 | 1806 |
| Sichuan | 四川省 | 1748 |
| Henan | 河南省 | 1496 |
| Beijing | 北京市 | 1113 |
| Ningxia | 宁夏回族自治区 | 950 |
| Gansu | 甘肃省 | 688 |
| Chongqing | 重庆市 | 434 |
| Jilin | 吉林省 | 401 |
| Yunnan | 云南省 | 394 |
| Guizhou | 贵州省 | 297 |
| InnerMongolia | 内蒙古自治区 | 285 |
| Guangxi | 广西壮族自治区 | 185 |

## Speakers per city-level regional subdialect

| Regional subdialect | Chinese | Speakers |
|---|---|---:|
| Ji-Lu | 冀鲁官话 | 4824 |
| Northeastern | 东北官话 | 4379 |
| Jiang-Huai | 江淮官话 | 3760 |
| Zhongyuan | 中原官话 | 3499 |
| Jiao-Liao | 胶辽官话 | 3133 |
| Southwestern | 西南官话 | 3058 |
| Beijing | 北京官话 | 2946 |
| Lan-Yin | 兰银官话 | 1638 |

Total: **34 cities, 18 province/municipality labels, 8 regional subdialects, and 27237 speakers**.
