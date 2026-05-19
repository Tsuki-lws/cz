# Skill Memory

| skill | count | applies_to |
| --- | ---: | --- |
| image_first_then_verify | 86 | other, location_landmark |
| entity_identification_then_verify | 34 | entity_name |
| concept_disambiguation | 31 | type_concept, visual_attribute |
| verify_date_year | 26 | date_year |
| answer_normalization | 26 | other, entity_name, location_landmark, country_region, date_year, type_concept |
| normalize_country_region | 17 | country_region |
| concise_answer | 9 | date_year, entity_name, location_landmark, type_concept, other |
| loop_breaker | 4 | type_concept, entity_name, other |
| tool_protocol_guard | 2 | entity_name, location_landmark |

## image_first_then_verify

- instruction: For landmarks, objects, posters, and visual entities, identify the visual target first; if uncertain, prefer a vision teacher or top-k candidates before finalizing.
- avoid: Do not guess a famous-looking landmark from a weak visual match.

## entity_identification_then_verify

- instruction: For person/book/author questions, first identify the entity, then answer the requested attribute such as appointer, disease, date, or role.
- avoid: Do not answer the attribute before the entity is reliable.

## concept_disambiguation

- instruction: For category, relation, style, family, or disease questions, choose the most specific concept that answers the exact wording.
- avoid: Do not return a broad superclass when the question expects a specific label.

## verify_date_year

- instruction: For date/year questions, verify the entity-year pair and output only the year/date span.
- avoid: Do not infer a year from visual style alone.

## answer_normalization

- instruction: Output the minimal answer span with no explanation, parentheses, citations, or extra qualifiers.
- avoid: Do not include surrounding sentences when a short answer is enough.

## normalize_country_region

- instruction: For country or region questions, verify origin carefully and normalize common aliases such as UK/United Kingdom and USA/United States.
- avoid: Do not confuse filming location, nationality, origin, and current location.

## concise_answer

- instruction: Use direct answer mode and keep the final answer under one short line unless the task asks otherwise.
- avoid: Do not write bullet points or analysis in benchmark answers.

## loop_breaker

- instruction: If reasoning repeats, stop exploring and provide the best supported final answer immediately.
- avoid: Do not repeat the same hypothesis or search wording.

## tool_protocol_guard

- instruction: When tools are enabled, use registered tool calls only; when tools are disabled, answer from image and knowledge without pseudo code.
- avoid: Do not print search commands or tool code as natural language.

