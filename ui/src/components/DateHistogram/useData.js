import { useEffect, useState } from "react"

export const useData = () => {
    const [articles, setArticles] = useState([]);
    const [reddit, setReddit] = useState([]);
    const [tweets, setTweets] = useState([]);
    useEffect(() => {
        fetch('data/article_dates.json').then((res) => {
            res.json().then(json => setArticles(json))
        })
        fetch('data/reddit_dates.json').then((res) => {
            res.json().then(json => setReddit(json))
        })
        fetch('data/tweet_dates.json').then((res) => {
            res.json().then(json => setTweets(json))
        })
    }, []);
    return { articles, reddit, tweets, loading: !articles.length || !tweets.length || !reddit.length }
}