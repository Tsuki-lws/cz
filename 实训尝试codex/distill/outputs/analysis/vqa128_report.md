# Eval Analysis: vqa_128

- count: 99
- accuracy: 0.2020
- norm_accuracy: 0.2323
- loose_accuracy: 0.2626

## Failure Counts

| failure_type | count |
| --- | ---: |
| IMAGE_MISREAD | 31 |
| CORRECT | 20 |
| ENTITY_MISIDENTIFIED | 13 |
| CONCEPT_ERROR | 12 |
| DATE_YEAR_ERROR | 10 |
| COUNTRY_REGION_ERROR | 7 |
| FORMAT_ERROR | 3 |
| PARTIAL_MATCH | 3 |

## Activated Skills

| skill | count |
| --- | ---: |
| image_first_then_verify | 31 |
| entity_identification_then_verify | 13 |
| concept_disambiguation | 12 |
| verify_date_year | 10 |
| normalize_country_region | 7 |
| answer_normalization | 6 |

## Examples

### index=0 DATE_YEAR_ERROR

- question: 图中人物是哪一年因肺结核去世的？
- gold: 1934年
- pred: 1936年
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=1 IMAGE_MISREAD

- question: What is the name of this bridge?
- gold: Bathampton Toll Bridge
- pred: Woolhope Bridge
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
- pred: 1994
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=5 IMAGE_MISREAD

- question: What is the name of this fortress?
- gold: Kalamita fortress
- pred: Fortress of St. Nicholas
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=6 CONCEPT_ERROR

- question: 在自然界中，这两个生物之间的关系是什么？
- gold: 互利共生
- pred: 共生关系
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=7 ENTITY_MISIDENTIFIED

- question: 图中这本书的作者死于什么疾病？
- gold: 霍乱
- pred: 心脏病
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=8 CONCEPT_ERROR

- question: 这张图片属于哪个类别？
- gold: 医学CT图像
- pred: 医学影像
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=9 ENTITY_MISIDENTIFIED

- question: Who appointed the man pictured here as Chief Justice of India in 1977?
- gold: Fakhruddin Ali Ahmed
- pred: Indira Gandhi
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=10 IMAGE_MISREAD

- question: Where is this vase sculpture located?
- gold: Bremen
- pred: Paris, France
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=11 ENTITY_MISIDENTIFIED

- question: 图中的人物是谁？
- gold: 艾维·莱德拜特·李（Ivy Ledbetter Lee）
- pred: 威廉·麦金莱
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=14 DATE_YEAR_ERROR

- question: 图中这个高坝是哪一年完工的？
- gold: 1970年
- pred: 1984
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=15 IMAGE_MISREAD

- question: 图中的塔2015年获得第56届什么奖项？
- gold: BCS（建筑承包商协会）奖
- pred: 日本建筑学会奖
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
- pred: The Seated Scribe
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
- pred: 本尼迪克特·康伯巴奇
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=21 IMAGE_MISREAD

- question: 这是哪个地标？
- gold: 埃及金字塔
- pred: 吉萨金字塔群
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=22 COUNTRY_REGION_ERROR

- question: From which country or region does this movie originate?
- gold: USA
- pred: United Kingdom
- skill: normalize_country_region
- better_strategy: Verify country/region aliases and normalize before final answer.

### index=23 IMAGE_MISREAD

- question: 图中这个湖是位于哪个市？
- gold: 赤峰市
- pred: 乌兰察布市
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=24 IMAGE_MISREAD

- question: Where is this location?
- gold: Marilao Church
- pred: Philippines
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=25 DATE_YEAR_ERROR

- question: In what year were the chemicals shown in the picture first discovered?
- gold: 1963
- pred: 1950
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=26 IMAGE_MISREAD

- question: Where is the cake located in relation to the camera?
- gold: Left side
- pred: in front of the camera
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=29 ENTITY_MISIDENTIFIED

- question: The person in the picture worked as a flight attendant until 1971. What's her name?
- gold: Jóhanna Sigurðardóttir
- pred: Barbara Castle
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=31 CONCEPT_ERROR

- question: 图中的植物属的拉丁学名是什么？
- gold: Raphidonema
- pred: 无法判断
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=32 IMAGE_MISREAD

- question: Which church is depicted in this image?
- gold: Church of Saint John the Baptist (Jihlava)
- pred: St. Martin's Church in Hohenberg, Bavaria
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

### index=33 ENTITY_MISIDENTIFIED

- question: Who is the developer of the science index in the picture?
- gold: 韦恩·帕默尔（Wayne Palmer）
- pred: McKee
- skill: entity_identification_then_verify
- better_strategy: Ask the vision teacher for top-k entity candidates before answering.

### index=34 CONCEPT_ERROR

- question: 这张图片属于哪个类别？
- gold: 医学CT图像
- pred: 医学影像
- skill: concept_disambiguation
- better_strategy: Disambiguate the visual concept before finalizing.

### index=35 DATE_YEAR_ERROR

- question: What year was the Lego set in the picture released?
- gold: 2002
- pred: 2005
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

