# Eval Analysis: memory

- count: 2
- accuracy: 0.0000
- norm_accuracy: 0.0000
- loose_accuracy: 0.0000

## Failure Counts

| failure_type | count |
| --- | ---: |
| DATE_YEAR_ERROR | 1 |
| IMAGE_MISREAD | 1 |

## Activated Skills

| skill | count |
| --- | ---: |
| image_first_then_verify | 1 |
| verify_date_year | 1 |

## Examples

### index=0 DATE_YEAR_ERROR

- question: 图中人物是哪一年因肺结核去世的？
- gold: 1934年
- pred: 1936
- skill: verify_date_year
- better_strategy: Verify date/year facts with text teacher and answer with only the year/date.

### index=1 IMAGE_MISREAD

- question: What is the name of this bridge?
- gold: Bathampton Toll Bridge
- pred: Woolhope Bridge
- skill: image_first_then_verify
- better_strategy: Route to the vision teacher first, then verify the named entity.

