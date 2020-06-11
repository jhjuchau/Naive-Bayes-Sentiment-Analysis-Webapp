# SeniorProject.SAE

NOTE: This project was developed in PyCharm. For best results, please import these files into a PyCharm project.

==Code Structure==
-GetTweets.py: main code hub, launches flask app, collects tweets based on topic
-SAEUtility.py: collection of data manipulation functions, called by GetTweets, SAEngine and TweetCleaning
-SAEngine.py: analysis code, called by GetTweets and SAEUtility 
-TweetCleaning.py: cleans tweet dataframes for use by analysis functions, called by GetTweets
-NaiveBayesTrainer.py: trains the NaiveBayes classifier, called by SAEngine

==Setup Instructions==
1. This project runs on Python. Python v3.7.2 and v3.8 were used during development.
2. The following libraries must be installed for proper use:
	-flask
	-GetOldTweets3
	-pandas
	-numpy
	-matplotlib
	-nltk
	-textblob
3. Run 'GetTweets.py' in a Python console window. The flask app should now be running at http://127.0.0.1:5000/
4. Open http://127.0.0.1:5000/ in a browser window. The app should be fully functional!
5. Any searches you perform will save local files, such as .csvs and .pngs, to the following directory:
	{project directory}/static/{topic searched}

NOTE: if any NLTK errors arise, you may have to install the 'punkt' NLTK tokenizers.
1. In the project .zip, you should see "punkt.zip"
2. Create a new folder under C:/{your username}/ called nltk_data
3. Unzip "punkt.zip" into that directory
