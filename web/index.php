<!DOCTYPE HTML>
<html>
<body>

<head>  
<link rel="stylesheet" type="text/css" href="common.css">
</head>

<?php
include "navigation.php"
?>

<article>
<div id="chartContainer"></div>
</article>

<nav>
<p>
Select test run to display <br>
</p>

<select name="run" id="runid" onchange="window.location='index.php?runid=' + this.value;">
<?php
$dir = "scorpiozero/";
$files = array_diff(scandir($dir), array('.', '..'));
$files = array_reverse($files);
foreach( $files as $i=>$f ) {
   if( $NET == "$dir$f" ) {
   	echo "\t<option value=$dir$f selected>Training run $f</option>\n";
   } else {
   	echo "\t<option value=$dir$f>Training run $f</option>\n";
   }
}
?>
</select>
<p>
<?php
$file = fopen("$NET/description.txt","r");

while(! feof($file))
  {
  echo fgets($file);
  }

fclose($file);
?>
</p>
</nav>

<script src="https://canvasjs.com/assets/script/canvasjs.min.js"></script>

<script>
window.onload = function () {
var chart = new CanvasJS.Chart("chartContainer", {
	animationEnabled: true,
	zoomEnabled: true,
	height: 600,
	theme: "dark1",
	axisX:{
 	        title: "Total games (K)",
		titleFontWeight: "bold",
		titleFontSize: 15,
		gridDashType: "dot",
		gridThickness: 2
 	},
        axisY:{
		title: "ELO",
		titleFontWeight: "bold",
		titleFontSize: 15,
		includeZero: true
	},
	toolTip: {
		shared: true
	},
	legend: {
		dockInsidePlotArea: true,
		cursor: "pointer",
		itemclick: toggleDataSeries
	},
	data: [{
		type: "rangeArea",
		markerSize: 0,
		name: "Strength",
		showInLegend: true,
		toolTipContent: "{x}<br><span style=\"color:#6D77AC\">{name}</span><br>Min: {y[1]} elo<br>Max: {y[0]} elo",
		dataPoints: [
<?php
$file = fopen("$NET/pltdata","r");

while(! feof($file))
  {
  echo fgets($file);
  }

fclose($file);
?>
		]
	}]
});
chart.render();

addAverages();

function addAverages() {
	var dps = [];
	for(var i = 0; i < chart.options.data[0].dataPoints.length; i++) {
		dps.push({
			x: chart.options.data[0].dataPoints[i].x,
			y: (chart.options.data[0].dataPoints[i].y[0] + chart.options.data[0].dataPoints[i].y[1]) / 2
		});
	}
	chart.options.data.push({
		type: "line",
		name: "Average",
		showInLegend: true,
		markerType: "triangle",
		markerSize: 0,
		yValueFormatString: "##.0 elo",
		dataPoints: dps
	});
	chart.render();
}

function toggleDataSeries(e) {
	if (typeof (e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
		e.dataSeries.visible = false;
	} else {
		e.dataSeries.visible = true;
	}
	e.chart.render();
}

}
</script>

</body>
</html>
