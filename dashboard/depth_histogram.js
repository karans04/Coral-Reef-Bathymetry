var margin = {top: 10, right: 30, bottom: 30, left: 60},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#IS2_depth_preds")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

const render = data => {
    svg.selectAll('rect').data(data)
        .enter().append('rect')
            .attr('width', 300)
            .attr('height', 300)
};

const preds_fn = './Moce.csv';
d3.csv(preds_fn).then(data => {
    keys = Object.keys(data[0]);
    data.forEach(d=>{
        if (d.median < 30 || d.median > 10){
            d.median = null;
        } else{
            d.median = +d.median;
        }
        
    });

    var max_depth = d3.max(data, d=>d.median),
        min_depth = d3.min(data, d=>d.median);
    console.log(max_depth, min_depth)

    var histogram = d3.histogram()
        .value(function(d) {    return d.median;   });
    var bins = histogram(data);
    console.log(bins);

});