# Digikala comments spam classification using deep merge neural network

## Dataset

| id  | title |                                                                     comment |  rate | verification_status |
| :-: | ----: | --------------------------------------------------------------------------: | ----: | ------------------: |
|  0  |     0 | کیفیت و حجم صدای عااااالی این محصول توی بازار اصلاااا پیدا نمیشه من کل ت... | 100.0 |                   0 |
|  2  |     2 |                    کارآیی به نظر من فقط برای کارای سبک و دیدن فیلم و مطا... |  60.0 |                   0 |
|  3  |     3 |          بررسی کمی و کیفی برای من بسیار مناسب و خریدش در شگفت انگیز حتما... |   0.0 |                   0 |
|  4  |     4 |            بسته بندی ضعیف ظاهر بامزه ای داره ولی عکسش شبیه خودش نیست جنس... |  60.0 |                   0 |

![Word count dist](/output/word_counts.png)
As it's visible in the figure, 92.53% of sentences have between 5 to 324 words.

### Data prepration

1000 most frequent words were selected and tokenized using keras Tokenizer.

```
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(comments + titles)
vec_comments = tokenizer.texts_to_sequences(comments)
vec_titles = tokenizer.texts_to_sequences(titles)
```

Comments rating were also normilized and mapped on the scale of 0 to 1.

```
ratings = ratings / 100
```

## Model

![model graph](/output/model_graph.png)
![loss](/output/tran-val-loss.png)

## Results

### Confusion Matrix

![confusion matrix](/output/confusion_matrix.png)

### Metrics

| Accuracy | F1 Score | Precision Score | Recall |
| :------: | -------: | --------------: | -----: |
|  0.882   |    0.537 |           0.411 |  0.773 |
