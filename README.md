
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
Using 



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
​
## Roadmap

- needs input from Amran/David
- Additional browser support

- Add more integrations


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


## Run Locally

Clone the project

```bash
  git clone https://github.com/tidehackathon/team-logically.git
```

Go to the project directory

```bash
  cd team-logically/ui
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


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


- Code? Languages etc

ReactJS (UI)

Python

Tools
-

## Data Science models

### Ground truth Elastic Index

Needs input from David

### Social media filtering based on coherence 

Needs input from David

### Semantic similarity Evidence Retrieval 

Needs input from David

### Entailment

Needs input from David

### Liklihood score

Needs input from David
