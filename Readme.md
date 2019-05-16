# Poem2Poem
Machine translation in poems domain preserving rhythm and rhyme

Inspired by DeepSpeare project:

https://arxiv.org/abs/1807.03491

https://github.com/jhlau/deepspeare

And by YSDA NLP course:

https://github.com/yandexdataschool/nlp_course

Also there is research paper by Marjan Ghazvininejad, Yejin Choi and Kevin Knight:

https://aclweb.org/anthology/N18-2011

But I have heard about it only after finishing this project


## Goal

This project is aimed at writing a program which can automatically translate poems in one language to another (English -> Russian as example) so that translated text has rhythm and rhyme. Ideally, the program would capture poetic meter and rhyme patterns from original poem and reproduce them in translated one. However, note that such perfectly translated version generally doesn't exist in nature at all. It means that we have to find some trade-off between rhythm, rhyme, fluency, adequacy, etc. of translation.


## Examples of results

| "The Road Not Taken"<br>by Robert Frost | Automatically generated translation<br>containing some rhyme and rhythm |
| :---: | :---: |
| Two roads diverged in a yellow wood,<br>And sorry I could not travel both<br>And be one traveler, long I stood<br>And looked down one as far as I could<br>To where it bent in the undergrowth; | две дороги поднялись на желтые дрова<br>и жаль что я мог не путешествовать<br>и одним путником я был со мной<br>и взглянула вниз вниз так как я смог<br>туда где он погнулся под букетом |

## Method

Model consists of two parts: translation model and poetic meter model. They are trained simultaneously using multitask approach. Both are implemented as Encoder-Decoder architecture with Attention mechanism. Despite this fact, actually they differ much. Idea of Meter model and most of code of its loss function is borrowed from DeepSpeare project.
Inference function is highly flexible and supports many modes that can be combined to each other independently:
- With or without rhythm
- With or without rhyme
- Type of rhyme
- Type of sampling and number of samples



## Data

There is no parallel corpus of poems large enough to be used for training such model. Four datasets were used to train and fine-tune this model:
- OpenSubtitles parallel English->Russian corpus
- Parallel corpus of songs parsed by me from the Internet special for this project (Sourse text is English, target text is Russian; Russian translation of song doesn't contain rhyme and rhythm)
- Parallel corpus of Russian classic poems translated to English by the model with the same architecture but trained on OpenSubtitles from Russian to English - Backtranslation approach.
- Small corpus of Shakespeare sonnets translated to Russian by Marshak

## How to use

This section will be added soon

## Acknowledgements

https://github.com/jhlau/deepspeare

https://github.com/yandexdataschool/nlp_course

https://github.com/IlyaGusev/rupo
