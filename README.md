Temporalia-TQIC
===============

Our work for the NTCIR-11 Temporalia challenge, Temporal Query Intent Classification sub-task: predicting the temporal orientation of search engine user queries.

![ScreenShot](http://www.cs.man.ac.uk/~filannim/projects/temporalia/gfx/temporalia.jpg)

We tackled the task as a machine learning classification problem, by proposing th use of temporal-oriented attributes specifically designed to minimise the sparsity of the models.

The best submitted run achieved 66.33% of accuracy, by correctly predicting the temporal orientation of 199 test instances out of 300. 

##Requirements

Python libraries:

* NLTK ([web page](http://www.nltk.org/))

Non-Python resources:

* ManTIME ([web page](https://github.com/filannim/ManTIME))
* Temporal NorMA ([web page](https://github.com/filannim/timex-normaliser)) [already in `/external`]

please update the `feature_extractor.py` file with the right paths.

##Contact
- Email: filannim@cs.man.ac.uk
- Web: http://www.cs.man.ac.uk/~filannim/
