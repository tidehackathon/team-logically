
# Project Title


## Name of the solution: 

NODDY Networked Disinformation detection system


Description of the solution itself: 

Enabling field commanders to quickly assess and mitigate digital influence operations, that can pose localised risks to operational integrity and force security and posture in theatre.

In today’s digital age, it is crucial that military operations take into account the potential risks posed by digital influence operations, which can threaten operational integrity and force security and posture in theater, both before and after deployment.

The solution…. is specifically designed to enable field commanders to rapidly and accurately assess and mitigate these risks, providing a critical layer of protection for friendly forces. Commanders can be confident in their ability to quickly identify and mitigate any digital influence operations that may pose a threat, ensuring that NATO forces are able to operate safely and effectively in any theater of operation and among local populations.

## User Experience:

Field commanders can upload a single piece of content or bulk upload a csv file to the platform, click analyse and be presented with a dashboard interface in hierarchy misinformation likelihood. 

With the most likely 'misinromation' content at the top the interface presents the user the option to further anaylse the content/claim and then approve or dismiss the claim. 

Why? This encourages the 'expert in the loop' approach, giving key account users the option to verify the AI/ML models output.

On selecting the action (Approve/Dismiss) the platform aims to provide this feedback back into the knowledge base to constantly train the models. 

Additionally the user can see a second page of already reviewed claims and the output of the platform + the expert review. 





## Overview of the product:
We take as input social media content and are able to cross reference it with a knowledge base, to be able to determine its likelyhood of disinoframtion.

The knowledge base was built 407 Ground truth articles. These articles span a timeframe of publishing which is relatively small in relation to the publishing of the social media posts. 

This, however, we show is not a problem necesserily ass in these use cases 'narratives' are repeated. Hence even a small time frame encompasses a lot of the knoweldge.

The limitation would be for posts published after, which would encompass a new development. However we appreciaate this would be easily resolved with additional data.

What has enabled us to provide such a robust solution is our use of a 'topic agnostic model', which doesnt need additional training on the use case/topic to be affective. Which has already demonstrated its accuracy in the field of mis/dis detection, see for example SNOPES. 



# Screenshots


## Iteration 1 - 21/02/2023
![](https://github.com/tidehackathon/team-logically/blob/main/iteration-1-21-02-23.gif)

## Iteration 2 - 22/02/2023
![](https://github.com/tidehackathon/team-logically/blob/main/iteration-2-22-02-23.gif)

## Iteration 3 - 23/02/2023
![](https://github.com/tidehackathon/team-logically/blob/main/iteration-3-23-02-23.gif)

## Iteration 4 - 23/02/2023
![](https://github.com/tidehackathon/team-logically/blob/main/iteration-4-23-02-23.gif)

# Architecture
## Architecture V1 - 22/02/2023
![](https://github.com/tidehackathon/team-logically/blob/main/architecture-v1.jpg)

## Architecture V2 - 23/02/2023
### Data science architecture 
![](https://github.com/tidehackathon/team-logically/blob/main/ds-architecture.jpg)
### Front end Architecture
![](https://github.com/tidehackathon/team-logically/blob/main/front-end-architecture.jpg)

## Installation

### UI
#### `cd ui`
#### `npm i` or `yarn`
Downloads and installs dependencies

#### `npm start` or `yarn start`
Runs the app in the development mode.
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Backend
#### `cd ai`
#### `conda create -n hackathon python=3.8`

#### `conda activate hackathon`

#### `pip install -r requirements.txt`

#### `pip install elasticsearch==7.17.9 elasticsearch-dsl==7.4.0`

#### `python app.py`

## Roadmap

### Front end
- Additional browser support
- User management
- sign up / Log in
- link to fact check claims
- Add more integrations
- Richer graphics


### Data Science
- Increase the dataset
- Adding more negative labels
- Include active learning approach - 'expert in the loop' to teach the mnodel
- Train models the specific use cases using specific annotated data
- implement additional signals based on tweet meta data - which has not been included in this iteration
- Enhance sentiment analysis
- Make it multi modal
- Multi lingual analysis
- Integrate more fact checking models
- Hate detection
- Entity extraction and analysis

## API Reference

_ needs input _

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


# Tooling 

What tools did we use? 
## Open source
### front end
bootstrap and reactstrap - for styling components
- https://reactstrap.github.io/
- https://getbootstrap.com/ 
highcharts - for graphs
- https://www.highcharts.com/blog/posts/frameworks/react/

react-wordcloud
- https://www.npmjs.com/package/react-wordcloud

react-toastify
- https://www.npmjs.com/package/react-toastify

keyword-extractor - a quick module that extracts keywords
- https://www.npmjs.com/package/keyword-extractor


### DS models: 
Non-proprietary 

SBERT: https://sbert.net/

XLNET Large: https://huggingface.co/docs/transformers/model_doc/xlnet

RAKE Keyword extraction algorithm
- used for pre processing
https://pypi.org/project/rake-nltk/

Elastic Search
- Building an index of the ground truth articles

https://www.elastic.co/

- username: elastic
- password: xjzoHBeQccXmR83VqwjOFqcH

uri: https://hackathon-deployment.es.us-east1.gcp.elastic-cloud.com

Optimised Keyword search using Elastic

PreProcessing step:
- NTLK Gensim library
- Emoji Library

To filter the text initially to remove unwanted characters that may intrude the understanding of the models.

- Code? Languages etc

ReactJS (UI)

Python

Tools
-

## Data Science models

### PreProcessing 

Using RAKE and other libraries (referenced in Tooling) we will remove unwanted characters from the text and keep only keywords.

### Ground truth Elastic Index

We index the 407 Guardian and New york times artcles inside elstic index, to be used for searches throughout the project.

### Social media filtering based on coherence 

We complete a search on the elastic index based on Keywords to filter the revelant content.

### Semantic similarity Evidence Retrieval 

After the intially filtering, we use a machine learning model (SBERT) to find the Ground truth claims closest to the tweet/reddit content.

Chosen because they are pre trained on a large amount of textual data, which has shows a formidable ability of undersrtanding semantic structure and meaning of sentences

### Entailment

Using another machine leardning model (XLNET LARGE) to determine whether the closest ground truth claims to support or contradict the social media claim and attribute a score.

Chosen because they are pre trained on a large amount of textual data, which has shows a formidable ability of undersrtanding semantic structure and meaning of sentences

### Liklihood score

the likelihood score is correlated to how many ground truth articles that contradictg the claim.

i.e.
Amount of contradict / total amount

