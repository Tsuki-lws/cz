# Eval Analysis: react_256

- count: 99
- accuracy: 0.2020
- norm_accuracy: 0.2626
- loose_accuracy: 0.3131

## Failure Counts

| failure_type | count |
| --- | ---: |
| IMAGE_MISREAD | 28 |
| CORRECT | 20 |
| ENTITY_MISIDENTIFIED | 10 |
| TOO_VERBOSE | 9 |
| CONCEPT_ERROR | 8 |
| DATE_YEAR_ERROR | 8 |
| FORMAT_ERROR | 6 |
| COUNTRY_REGION_ERROR | 5 |
| PARTIAL_MATCH | 5 |

## Activated Skills

| skill | count |
| --- | ---: |
| image_first_then_verify | 28 |
| answer_normalization | 11 |
| entity_identification_then_verify | 10 |
| concise_answer | 9 |
| concept_disambiguation | 8 |
| verify_date_year | 8 |
| normalize_country_region | 5 |

## Examples

### index=0 DATE_YEAR_ERROR

- question: 图中人物是哪一年因肺结核去世的？
- gold: 1934年
- pred: 1938
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=1 IMAGE_MISREAD

- question: What is the name of this bridge?
- gold: Bathampton Toll Bridge
- pred: Painswick Bridge
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=2 COUNTRY_REGION_ERROR

- question: 图中运动起源于哪个国家？
- gold: 法国
- pred: 英国
- skill: normalize_country_region
- better_strategy: Verify country/region aliases and normalize before final answer.

### index=3 COUNTRY_REGION_ERROR

- question: What country or region is this movie originated from?
- gold: UK
- pred: United Kingdom
- skill: normalize_country_region
- better_strategy: Verify country/region aliases and normalize before final answer.

### index=4 DATE_YEAR_ERROR

- question: 图中的报纸是哪一年正式建立的？
- gold: 1998
- pred: 1992
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=5 IMAGE_MISREAD

- question: What is the name of this fortress?
- gold: Kalamita fortress
- pred: Chufut-Kale
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=6 FORMAT_ERROR

- question: 在自然界中，这两个生物之间的关系是什么？
- gold: 互利共生
- pred: 互利共生（清洁共生关系）
- skill: answer_normalization
- better_strategy: Normalize aliases and strip units/punctuation before scoring or training.

### index=7 ENTITY_MISIDENTIFIED

- question: 图中这本书的作者死于什么疾病？
- gold: 霍乱
- pred: 肺结核
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=8 CONCEPT_ERROR

- question: 这张图片属于哪个类别？
- gold: 医学CT图像
- pred: 医学影像（腹部CT扫描）
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=10 IMAGE_MISREAD

- question: Where is this vase sculpture located?
- gold: Bremen
- pred: The vase sculpture is located at the Fontaine des Quatre-Saisons in the Jardin des Plantes, Paris, France.
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=11 ENTITY_MISIDENTIFIED

- question: 图中的人物是谁？
- gold: 艾维·莱德拜特·李（Ivy Ledbetter Lee）
- pred: 托马斯·伍德罗·威尔逊（Thomas Woodrow Wilson）
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=12 PARTIAL_MATCH

- question: 图中这个概念最初是用来描述哪一种物品的运输系统？
- gold: 牛奶
- pred: 牛奶运输系统
- skill: answer_normalization
- better_strategy: Train the student to output only the minimal answer span.

### index=13 IMAGE_MISREAD

- question: 图中的这个桥隧全长多少千米（回答到整数即可）？
- gold: 55
- pred: 22
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=14 DATE_YEAR_ERROR

- question: 图中这个高坝是哪一年完工的？
- gold: 1970年
- pred: 1935
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=15 IMAGE_MISREAD

- question: 图中的塔2015年获得第56届什么奖项？
- gold: BCS（建筑承包商协会）奖
- pred: 阿卡普尔科国际建筑奖
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=16 DATE_YEAR_ERROR

- question: 图中这个人在哪一年成为威斯康星州的参议员？
- gold: 1946年
- pred: 1947年
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=17 CONCEPT_ERROR

- question: What is the title of this artwork?
- gold: Letizia Ramolino Bonaparte
- pred: Psyche Revived by Cupid's Kiss
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=18 CONCEPT_ERROR

- question: The third nebula in the first row from left to right was first discovered by which astronomer?
- gold: 卡罗琳·赫歇尔（Caroline Lucretia Herschel）
- pred: William Herschel
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=20 IMAGE_MISREAD

- question: 这个人是谁？
- gold: Benedict Cumberbatch
- pred: 本尼迪克特·康伯巴奇（Benedict Cumberbatch）
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=21 IMAGE_MISREAD

- question: 这是哪个地标？
- gold: 埃及金字塔
- pred: 吉萨金字塔群（Giza Pyramids），位于埃及吉萨高原，是古埃及第四王朝法老胡夫、哈夫拉和孟卡拉的陵墓，也是世界七大奇迹中唯一现存的地标。
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=22 COUNTRY_REGION_ERROR

- question: From which country or region does this movie originate?
- gold: USA
- pred: United Kingdom (co-production with the United States)
- skill: normalize_country_region
- better_strategy: Verify country/region aliases and normalize before final answer.

### index=23 IMAGE_MISREAD

- question: 图中这个湖是位于哪个市？
- gold: 赤峰市
- pred: 巴彦淖尔市
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=24 TOO_VERBOSE

- question: Where is this location?
- gold: Marilao Church
- pred: Based on the image, this appears to be a church in a tropical or subtropical region, possibly in Latin America or Southeast Asia. The pink facade and white dome are distinctive features that could help identify the specific location. More information is needed to determine the exact location.
- skill: concise_answer
- better_strategy: Use direct answer mode and reduce max_tokens for benchmark inference.

### index=25 TOO_VERBOSE

- question: In what year were the chemicals shown in the picture first discovered?
- gold: 1963
- pred: The image shows the chemical structure of **chitin oligosaccharide**, specifically a trimer of N-acetylglucosamine units linked by β(1→4) glycosidic bonds — this is a fragment of **chitin**, a polysaccharide found in the exoskeletons of arthropods and cell walls of fungi.

However, the question asks: “In what year were the chemicals shown in the picture first discovered?”

The monomer unit is **N-acetylglucosamine (GlcNAc)**, and the polymer is **chitin**.

- **Chitin** was first isolated and described in **1811** by **Henri Braconnot**, who obtained it from crab shells.
- **N-acetylglucosamine** was later identified as the monomer of chitin. It was first synthesized and characterized in the mid-19th century, but its identification as the building block of chitin came with the structural elucidation of chitin in the 20th century.

But since the image depicts a **chitin oligomer** (a short chain of GlcNAc units), and chitin itself is the primary substance represented, we should
- skill: concise_answer
- better_strategy: Use direct answer mode and reduce max_tokens for benchmark inference.

### index=26 IMAGE_MISREAD

- question: Where is the cake located in relation to the camera?
- gold: Left side
- pred: in front of the camera
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=29 ENTITY_MISIDENTIFIED

- question: The person in the picture worked as a flight attendant until 1971. What's her name?
- gold: Jóhanna Sigurðardóttir
- pred: Margaret Hodge
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=30 TOO_VERBOSE

- question: 图中这个植物属于哪一个植物目？
- gold: 天门冬目
- pred: 根据图像中植物的形态特征：

- 花朵为鲜艳的蓝紫色，呈星形，6枚花瓣状花被片；
- 花被片上有明显的纵向条纹；
- 雄蕊6枚，花丝细长，花药黄色；
- 植株具有肉质叶片（背景中可见灰绿色肉质叶），符合石蒜科或葱科某些属的特征；
- 整体花序和花型与**葱科（Asparagaceae）**下的**蓝星花属（Triteleia）**或**百子莲属（Agapanthus）**等相似，但更贴近**葱科**中一些小型球根花卉。

然而，从专业分类角度，这类花型、6基数花部结构、雄蕊6枚、花被片条纹明显，且常伴肉质叶的植物，最可能属于：

> **石蒜科（Amaryllidaceae）**

特别是**葱属（Allium）**或**蓝星花属（Triteleia）**，但蓝星花属现已被归入**石蒜科**。

进一步确认：图中植物极可能是 **Triteleia laxa**（蓝星花），它属于：
- skill: concise_answer
- better_strategy: Use direct answer mode and reduce max_tokens for benchmark inference.

### index=31 CONCEPT_ERROR

- question: 图中的植物属的拉丁学名是什么？
- gold: Raphidonema
- pred: 图中展示的是珊瑚化石，其所属植物（更准确地说是刺胞动物）的拉丁学名是 Scleractinia（石珊瑚目）。
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=32 IMAGE_MISREAD

- question: Which church is depicted in this image?
- gold: Church of Saint John the Baptist (Jihlava)
- pred: St. Michael's Church in Hohenberg, Germany
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=33 ENTITY_MISIDENTIFIED

- question: Who is the developer of the science index in the picture?
- gold: 韦恩·帕默尔（Wayne Palmer）
- pred: Wayne Palmer
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

