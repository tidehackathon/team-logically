import React from 'react'
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import moment from 'moment';
import articles from './article_dates.json';
import reddit from './reddit_dates.json';
import tweets from './tweet_dates.json';

export const DateHistogram = () => {
    const options = {
        chart: {
            zoomType: 'x'
        },
        title: {
            text: ''
        },
        xAxis: {
            type: 'datetime'
        },
        yAxis: {
            title: {
                text: 'Number of content'
            }
        },
        series: [
            {
                data: combineDates(articles),
                name: 'Articles'
            },
            {
                data: combineDates(reddit),
                name: 'Reddit'
            },
            {
                data: combineDates(tweets),
                name: 'Twitter'
            }
        ]
    }
    return <div>
        <h2>Sample data</h2>
        <HighchartsReact
            highcharts={Highcharts}
            options={options}
        />
    </div>
};

const combineDates = (data) => {
    let ret = [];
    for (const date of data) {
        const startOfDay = moment(new Date(date)).startOf('day').valueOf();
        const foundIndex = ret.findIndex(item => item.x === startOfDay);
        if (foundIndex !== -1) {
            ret[foundIndex].y += 1;
        } else {
            ret.push({ x: startOfDay, y: 1 })
        }
    }
    return ret.sort((a, b) => b.x - a.x);
}
