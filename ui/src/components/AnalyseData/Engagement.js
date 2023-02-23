import React from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import moment from 'moment';

export const Engagement = ({ data }) => {
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
                text: 'Engagement'
            }
        },
        series: [
            {
                data: combineDates(data.filter(item => item.percentage > 40)),
                name: 'Likely disinformation',
                color: '#F84367'
            },
            {
                data: combineDates(data.filter(item => item.percentage < 40)),
                name: 'Unlikely disinformation',
                color: '#13BA9C'
            }
        ]
    }
    return <div>
        <h3>Engagement</h3>
        <HighchartsReact
            highcharts={Highcharts}
            options={options}
        />
    </div>
};

const combineDates = (data) => {
    let ret = [];
    for (const item of data) {
        const startOfDay = moment(new Date(item.date)).startOf('day').valueOf();
        const foundIndex = ret.findIndex(retItem => retItem.x === startOfDay);
        if (foundIndex !== -1) {
            ret[foundIndex].y += item.engagement;
        } else {
            ret.push({ x: startOfDay, y: item.engagement })
        }
    }
    return ret.sort((a, b) => b.x - a.x);
}

