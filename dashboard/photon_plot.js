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

const reef_preds_path = 'data/Moce/Output/Data_Cleaning/Data_plots/Moce_ATL03_20181210123712_11130108_002_01_gt1r_plots.csv';
    path_split = reef_preds_path.split('/'),
    fn = path_split[path_split.length -1];
    
    const reef_path = path_split.slice(0,2).join('/'),
        metadata = reef_path + '/' + 'ICESAT_metadata.json';

    const h5_fn = fn.split('_').slice(1,6).join('_') + '.h5',
        laser = fn.split('_')[6];


d3.csv(reef_preds_path).then(data => {

    var latitude = [],
        photon_depth = [],
        predicted_depth = [],
        photon_len = data.length;

    for (i = 0; i < photon_len; i += 1){
        latitude.push(parseFloat(data[i]['Latitude']));
        photon_depth.push(parseFloat(data[i]['Photon_depth']));
        predicted_depth.push(parseFloat(data[i]['Predicted_depth']));
    }
    var max_y = Math.max.apply(null,photon_depth.filter(x=>!isNaN(x))),
        min_y = Math.min.apply(null,photon_depth.filter(x=>!isNaN(x))),
        min_x = Math.min.apply(null,latitude.filter(x=>!isNaN(x))),
        max_x = Math.max.apply(null,latitude.filter(x=>!isNaN(x)));
    // create x scale
    var xScale = d3.scaleLinear()
        .domain([min_x, max_x])
        .range([ 0, width ]);
    // Add y scale
    var yScale = d3.scaleLinear()
        .domain([min_y, max_y])
        .range([ height, 0]);
    // create axis objects
    var xAxis = d3.axisBottom(xScale)
        .ticks(5, "s");
    var yAxis = d3.axisLeft(yScale);

    // draw axis
    var gX = svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);
    var gY = svg.append("g")
        .call(yAxis);

    // draw axis label
    svg.append("text")
        .attr("class", "x label")
        .attr("text-anchor", "end")
        .attr("x", width - (width/2))
        .attr("y",height+ 30)
        .text("Latitude");

    svg.append("text")
        .attr("class", "y label")
        .attr("text-anchor", "end")
        .attr("y",-30)
        .attr("x", -height + (height /2))
        .attr("transform", "rotate(-90)")
        .text("Depth (m)");

    // add ICESAT dots
    var points = svg.append('g')
        .selectAll("dot")
        .data(data)
        .enter()
        .append("circle")
            .attr("cx", function (d) { return xScale(d.Latitude); } )
            .attr("cy", function (d) { return yScale(d.Photon_depth); } )
            .attr("r", 0.5)
            .style("fill", "#008000");
    // add photon pred dots
    var points2 = svg.append('g')
        .selectAll("dot")
        .data(data.filter(function(d) { return d.Predicted_depth != '';}))
        .enter()
        .append("circle")
            .attr("cx", function (d) { return xScale(d.Latitude); } )
            .attr("cy", function (d) { return yScale(d.Predicted_depth); } )
            .attr("r", 1.5)
            .style("fill", "#FF0000");

    d3.json(metadata).then(d => {
        sea_level_func = d[h5_fn]['sea_level_func'][laser];
        tide = d[h5_fn]['tide'];

        function createPoints(a,b,c,rangeX,step){
            return Array.apply(null,Array((rangeX[1]-rangeX[0])/step|0 + 1))
            .map(function(d,i){
                    var x = rangeX[0]+i*step;
                    return a * x * x + b * x + c;
            })
        }
        var a = sea_level_func[0],
            b = sea_level_func[1],
            c = sea_level_func[2],
            step = (max_x - min_x)/25,
            points = [];
        
        var y = createPoints(a,b,c,[min_x,max_x],step);
        var total = 0;
        for(var i = 0; i < y.length; i++) {
            total += y[i];
        }
        var mean_sea = total / y.length;
        for(var i = 0; i < y.length; i++) {
            y[i] -= mean_sea;
            points.push({'x':min_x+i*step, 'y':y[i]});
        }      
        console.log(points)
        var valueline = d3.line()
            .x(function(d) { return xScale(d.x); })
            .y(function(d) { return yScale(d.y); });
        // Add the valueline path.
        svg.append("path")
            .attr("class", "line")
            .attr("d", valueline(points))
            .attr("stroke", "blue")
            .attr("stroke-width", 2)
            .attr("fill","#008000");

    });

    
})