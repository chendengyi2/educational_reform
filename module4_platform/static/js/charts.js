/**
 * ECharts图表通用配置
 */

// 深色主题配置
var darkThemeOption = {
    backgroundColor: 'transparent',
    textStyle: {
        color: '#ccc'
    },
    title: {
        textStyle: { color: '#eee' }
    },
    legend: {
        textStyle: { color: '#aaa' }
    },
    tooltip: {
        backgroundColor: 'rgba(0,0,0,0.7)',
        borderColor: '#333',
        textStyle: { color: '#fff' }
    },
    categoryAxis: {
        axisLine: { lineStyle: { color: '#444' } },
        axisLabel: { color: '#aaa' },
        splitLine: { lineStyle: { color: '#333' } }
    },
    valueAxis: {
        axisLine: { lineStyle: { color: '#444' } },
        axisLabel: { color: '#aaa' },
        splitLine: { lineStyle: { color: '#333' } }
    }
};

// 调色板
var chartColors = [
    '#5470c6', '#91cc75', '#ee6666', '#fac858', '#73c0de',
    '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#48b8d0'
];

/**
 * 通用柱状图配置
 */
function createBarOption(data) {
    return {
        tooltip: { trigger: 'axis' },
        legend: { data: data.legend, textStyle: { color: '#ccc' } },
        xAxis: { type: 'category', data: data.categories, axisLabel: { color: '#aaa' } },
        yAxis: { type: 'value', axisLabel: { color: '#aaa' } },
        series: data.series.map(function(s, i) {
            return {
                name: s.name, type: 'bar', data: s.data,
                itemStyle: { color: chartColors[i % chartColors.length] }
            };
        })
    };
}

/**
 * 通用折线图配置
 */
function createLineOption(data) {
    return {
        tooltip: { trigger: 'axis' },
        legend: { data: data.legend, textStyle: { color: '#ccc' } },
        xAxis: { type: 'category', data: data.dates, axisLabel: { color: '#aaa' } },
        yAxis: { type: 'value', axisLabel: { color: '#aaa' } },
        series: data.series.map(function(s, i) {
            return {
                name: s.name, type: 'line', data: s.data, smooth: true,
                lineStyle: { width: 2, color: chartColors[i % chartColors.length] },
                itemStyle: { color: chartColors[i % chartColors.length] }
            };
        })
    };
}

/**
 * 通用饼图配置
 */
function createPieOption(data) {
    return {
        tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
        legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { color: '#ccc' } },
        series: [{
            type: 'pie',
            radius: data.radius || ['40%', '70%'],
            center: data.center || ['40%', '50%'],
            data: data.items,
            label: { color: '#ccc' },
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0,0,0,0.5)'
                }
            }
        }]
    };
}

/**
 * 通用散点图配置
 */
function createScatterOption(data) {
    return {
        tooltip: { trigger: 'item' },
        legend: { data: data.legend, textStyle: { color: '#ccc' } },
        xAxis: {
            name: data.xName || '',
            axisLabel: { color: '#aaa' },
            splitLine: { lineStyle: { color: '#333' } }
        },
        yAxis: {
            name: data.yName || '',
            axisLabel: { color: '#aaa' },
            splitLine: { lineStyle: { color: '#333' } }
        },
        series: data.series.map(function(s, i) {
            return {
                name: s.name, type: 'scatter', data: s.data,
                symbolSize: s.size || 8,
                itemStyle: { color: chartColors[i % chartColors.length] }
            };
        })
    };
}

/**
 * 通用雷达图配置
 */
function createRadarOption(data) {
    return {
        tooltip: {},
        legend: { data: data.legend, textStyle: { color: '#ccc' } },
        radar: {
            indicator: data.indicators,
            axisName: { color: '#aaa' },
            splitArea: {
                areaStyle: {
                    color: ['rgba(58,123,213,0.1)', 'rgba(58,123,213,0.05)']
                }
            }
        },
        series: [{
            type: 'radar',
            data: data.series.map(function(s, i) {
                return {
                    name: s.name,
                    value: s.value,
                    areaStyle: { opacity: 0.2 },
                    lineStyle: { width: 2 }
                };
            })
        }]
    };
}
