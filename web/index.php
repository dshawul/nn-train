<!DOCTYPE HTML>
<html>

<head>  
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>

<body>

<header>
<h2>ScorpioZero Neural Network Training.</h2>
</header>

<nav>
<p>
Welcome to the ScorpioZero training server. <br>
</p>

<?php
include "common.php";
?>

<p>
<a href="http://scorpiozero.ddns.net/users.php">List of users</a> <br>
<a href="http://scorpiozero.ddns.net/contrib.php">Contributions of users</a> <br>
<a href="http://scorpiozero.ddns.net/work.php">Active training runs</a> <br>
<a href="http://scorpiozero.ddns.net/scorpiozero">Download networks and training data</a> <br>
<a href=<?php echo "http://scorpiozero.ddns.net/matches.php?runid=".$NET; ?> >View matches between networks</a> <br>
</p>

<p>
Github pages. <br>
</p>

<a href="https://github.com/dshawul/nn-dist">Client for training</a> <br>
<a href="https://github.com/dshawul/nn-train">Server for training</a> <br>
<a href="https://github.com/dshawul/nn-probe">Probing code</a> <br>

<p>
Discussion. <br>
</p>

<a href="https://groups.google.com/forum/#!forum/scorpiozero">Discussion forum</a> <br>

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

</nav>

<article>
<div id="chartContainer"></div>
</article>

<nav>
<p>
Latest run still with 2x32 net and 512 games per net
with a sliding window of  15 nets. <br>
</p>

<p>
Also initial learning rate is 0.15 as opposed to 0.01 in run 1.
The previous net had problems capturing peice values even until 600k games. <br>
</p>
</nav>

<footer>
<p>..</p>
</footer>

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
