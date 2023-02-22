import React from 'react'
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import moment from 'moment';
import { useData } from './useData';
import { Spinner } from 'reactstrap';

export const DateHistogram = () => {
    const { articles, reddit, tweets, loading } = useData()
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
    return <div className="position-relative mt-2">
        {loading && <div className="absolute-center"><Spinner color="primary" /></div>}
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
