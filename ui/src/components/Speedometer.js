import React from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import HighchartsMore from 'highcharts/highcharts-more';

HighchartsMore(Highcharts);

export const Speedometer = ({ value = 50 }) => {
    const thickness = 6;
    const options = {
        chart: {
            height: 50,
            width: 50,
            type: 'gauge',
            plotBackgroundColor: null,
            plotBackgroundImage: null,
            plotBorderWidth: 0,
            plotShadow: false,
            // height: '80%'
        },
        title: {
            text: ''
        },
        pane: {
            startAngle: -90,
            endAngle: 90,
            background: null,
            center: ['50%', '50%'],
            size: '170%'
        },
        yAxis: {
            min: 0,
            max: 100,
            tickPosition: 'inside',
            tickLength: thickness,
            tickWidth: 1,
            minorTickInterval: null,
            labels: {
                enabled: false
            },
            plotBands: [{
                from: 0,
                to: 20,
                color: '#13BA9C',
                thickness
            }, {
                from: 20,
                to: 40,
                color: '#88C458',
                thickness
            }, {
                from: 40,
                to: 60,
                color: '#FCCD13',
                thickness
            }, {
                from: 60,
                to: 80,
                color: '#FA883D',
                thickness
            }, {
                from: 80,
                to: 100,
                color: '#F84367',
                thickness
            }]
        },
        series: [{
            name: '',
            data: [value],
            dataLabels: {
                enabled: false
            },
            dial: {
                radius: '80%',
                backgroundColor: '#295F87',
                baseWidth: 6,
                baseLength: '0%',
                rearLength: '0%'
            },
            pivot: {
                backgroundColor: '#295F87',
                radius: 3
            }
        }],
        tooltip: {
            enabled: false
        }
    }
    return <HighchartsReact
        highcharts={Highcharts}
        options={options}
    />
};
