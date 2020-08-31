### Which Party in the USA?
## by Anne Barry

**Background and Essential Question**
Which political party comes to mind with the words “federal reserve”? “Social security”? “God Bless America"?  Over the last two weeks, we heard many speeches given at both the Democratic and Republican National Conventions.  Pundits don’t always agree, but what they did agree on was that each party’s convention differed in theme and tone, sometimes to a dangerous degree.  [“Nikki Haley defends darker tone of Republican National Convention](https://www.cbsnews.com/news/trump-nikki-haley-republican-national-convention/),” said CBSNews, echoing Fox News’ [“Sen Amy Klobuchar defends tone of Democratic National Convention”](https://www.youtube.com/watch?v=u2xBkRSmVJU) the previous week.  To each news outlet, one of the parties had distinguished itself by tone, which is certainly in the eye of the beholder.  How much do actual words play in distinguishing one party from another?  With a model that pays no attention to context or phrasing, can we still tell the difference between the speech of a Republican and that of a Democrat? What can we learn from the terms that define partisan speech? In this project, I begin to answer this question through natural language processing.

**The data:**

I acquired the data through two sources: [TheGrammarLab](http://www.thegrammarlab.com/?nor-portfolio=corpus-of-presidential-speeches-cops-and-a-clintontrump-corpus) and [Debates.org](http://debates.org).  I downloaded 486 .txt files from the former and used BeautifulSoup to scrape an additional 25 debate transcripts from the latter.  The .txt files included speeches from all presidents through Obama, but I only used Presidents Hoover through Obama because those presidents are most closely aligned with the modern Democratic and Republican parties.  The debate transcripts include most Presidential debates from 1960 to 2012 as well as Vice Presidential debates from 1984 to 2016.  Though I found sources for them, President Trump's speeches were excluded based on the perception that his prose differs greatly from any modern Republican (though there's more on that later). Speeches include inaugural addresses, press conferences, state of the union addresses, fireside chats, among others.
After the data was imported, my pipeline parsed the files/html so that the text could be extracted.  This was especially challenging for the debates, as multiple speakers were in the same transcript and I needed to split the text according to speaker.  Once I compiled my data into a dataframe, I began EDA.<br>

![The data](/img/df_head.png)

**Exploratory Data Analysis:**

The corpus of speeches breaks down as follows:

**Number of Democrats**: 16, including 7 presidents and an additional 9 candidates who appeared in debates

**Number of Republicans**: 15, including 7 presidents and an additional 8 candidates who appeared in debates

**Number of speeches by Democrats**: 322
**Number of speeches by Republicans**: 227

**Most speeches**: Lyndon B. Johnson (71), Ronald Reagan (60), Barack Obama (54)
**Fewest speeches**: Dwight D. Eisenhower (6), Gerald Ford (17), Harry S. Truman (19)

![Speeches per Speaker](/img/speeches_by_pres.png)

**Most press conferences**: Lyndon B. Johnson (23) followed distantly by several others with 1 each

**Unique words in corpus**: 57,629 (1,928,737 words)

**Median text length**: 3,176 words (21.17 minutes, speaking at a rate of 150 words/minute)

**Lengthiest text**: Lyndon B. Johnson’s 1967 State of the Union Address (14,176 words)

**Shortest text**: Franklin D. Roosevelt’s “Message to Congress Requesting War Declarations with Germany and Italy” on December 11, 1944 (144 words)

Though the corpus included about 100 more speeches by Democrats than by Republicans, the distribution of text length between the parties was roughly the same, showing that I didn't have a disproportionately large number of long speeches from one party or the other.

![ ](/img/word_count_by_party.png)

The corpus is by no means exhaustive; Nixon, with 27 total speeches in the corpus (including his famous Kennedy debates), is known to have given 24 Oval Office addresses, which implies that I am missing a large number of speeches overall.


**Modeling**

Because of the large vocabulary but small sample size, I chose to use a Multinomial Naive Bayes classifier to build a model that would predict whether a given text was delivered by a Republican or a Democrat.  This model is excellent in classifying different types of texts but operates on one major assumption: the probability that a word is included in a text is independent of the presence of other words in the text.  This means that phrasing as well as context are not detected by this model.

I used scikit-learn’s TFIDF Vectorizer with a list of stop words compiled by running the model several times and looking for words that are irrelevant to politics, such as the word “applause,” which showed up in many debate transcripts.  The TFIDF Vectorizer multiplies a word’s term frequency by the inverse of its document frequency, providing values that reflect a term’s importance as a balance between the term’s frequency within an individual speech and its rarity across all speeches.  Therefore, a term like “America”, used many times within speeches (term frequency) but also used in most speeches (document frequency), it ends up being a weaker predictor of the speaker’s party than terms infrequently used terms like “voting rights act”, but are most commonly used by one party over the other.

I evaluated the model by its accuracy, which is the total number of correct predictions divided by the total number of predictions.  Recall, defined as the proportion of true positives divided by the sum of true positives and false negatives, was also calculated for each class as a means of evaluating the model.  Initial tests showed that accuracy was around 70%, recall for Democrats was consistently above 90% while the recall for Republicans hovered around 30%.

In a Multinomial Bayes model, hyperparameters to optimize include the number of ngrams used as features in the model (ngrams are comprised of n consecutive words in a text), the length of the ngrams, and the threshold for classifying something as a “positive” result (in this case, the model defaulted to classifying “Democrat” as positive).

By examining the effects of changes in these hyperparameters, I determined that the optimal model included 1500 features, trigrams only, and a 54% threshold for classifying a text as Democratic. The accuracy ranged from around 78% to 86%.  Recall for was 92.5% and 71.9% for Democrats and Republicans, respectively.

To determine the best number of features, I plotted overall accuracy as well as recall for both parties as the number of features changed. Recall for Republicans decreased drastically as the number of features increased, so I chose 1500 features, which seemed like a point where there as a reasonable tradeoff between the recall rates for each party.

![ ](/img/accuracy_vs_recall.png)

To determine the best threshold, I plotted accuracy vs. threshold and saw that a threshold of .54 would give me the best results.  It makes sense to have a threshold greater than .50 because my initial model was misclassifying Republicans as Democrats; now with the threshold higher, more Republicans would be properly classified.

![ ](/img/accuracy.png)

The ROC curve for my model captures an area under the curve of .918 and the breakdown of the confusion matrix for my model is below.

![ ](/img/roc.png)

![ ](/img/confusion.png)


I also optimized a random forest classifier with a random search and achieved a consistent score of around 80%, using 1-3ngrams, a.54 threshold, 600 estimators, and a max depth of 115, but found that the Bayes model outperformed it.


Results:

Once I ran the multinomial Bayes model, I created a list of the tri-grams with the top 50 log probabilities (a measure of a string’s predictive strength) for each party.  Then, I cross-referenced the lists to find the trigrams that were unique to each party’s top 50 list.

Top Democratic Indicators:
```['health care reform', 'third world war', 'nuclear arms race', 'strength united states', 'executive branch government', 'voting rights act', 'president members congress', 'social security medicare', 'soldiers sailors marines', 'hundred million dollars', 'support american people', 'spread nuclear weapons', 'people back work', 'policy united states', 'north viet nam', 'government south vietnam', 'united states nation', 'world ever known', 'health care costs', 'chiang kai shek', 'god bless united', 'constitution united states', 'million new jobs', 'test ban treaty', 'general de gaulle', 'civil rights bill', 'united states government', 'bless united states', 'peace loving nations', 'people south vietnam', 'bless god bless', 'god bless god', 'martin luther king', 'middle class families', 'joint chiefs staff', 'second world war', 'south viet nam']```

Top Republican Indicators:

```['vice president bush', 'thousand points light', 'report state union', 'mutual self help', 'balanced budget amendment', 'united states military', 'much remains done', 'people republic china', 'forces united states', 'gross national product', 'members united states', 'department homeland security', 'since world war', 'building loan associations', 'peace among nations', 'federal reserve system', 'american people know', 'god bless thank', 'strategic defense initiative', 'god bless america', 'peace middle east', 'united states seek', 'high interest rates', 'federal reserve banks', 'federal farm board', 'new world order', 'instrument national policy', 'intermediate range nuclear', 'states soviet union', 'special session congress', 'line item veto', 'general secretary gorbachev', 'united states soviet', 'go forward together', 'nations security council', 'united nations security', 'united states congress']```

Included in the Democrats' list are many terms that refer to social programs or civil rights, such as ```health care reform```, ```social security medicare```, and ```civil rights bill```, whereas the Repblicans have a fiscal focus, using phrases like ```federal reserve system```, ```balanced budget amendment```, and ```high interest rates```. There also seems to be more of a military inclination on the Republican side, and it is interesting to note the presence of the spelling "Viet Nam" on the Democrat side, as this is how it is transcribed for most of Kennedy's and Johnson's speeches.

Below is a breakdown of how often certain trigrams are used per 100,000 words in each party (with two cheeky examples at the end).

![ ](/img/words.png)


Lastly, I ran Trump's 2016 campaign speeches as well as his 2020 Republican National Convention speech through the model to see what party he would be classified as.  I expected the model to have trouble distinguishing a party for him as his word choice strongly diverges from politicians that have come before him.  For his twelve 2016 speeches, he was classified as a Republican only 50% of the time. Surprisingly, his 2020 RNC speech classified him as a Democrat, with a probability of .65.

**Conclusion and Next Steps**

My model provides insight in regards to the topics that each party talks about, but I would also like to explore sentiment analysis in the future.  Additionally, I am not certain why more features reduces recall for Republicans, but one thought is that Republicans may share only a small set of phrases between them, whereas Democrats may use a broader set of phrases, but that would be worth exploring.