<!DOCTYPE HTML>
<html>
<head>  
<script>
window.onload = function () {

var chart = new CanvasJS.Chart("chartContainer", {
	animationEnabled: true,
	theme: "light2",
	axisX:{
 	        title: "Total games",
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
	data: [{        
		type: "line",       
		lineColor: "red",
		dataPoints: [

<?php
$file = fopen("scorpiozero/nets-1/pltdata","r");

while(! feof($file))
  {
  echo "{ ".fgets($file)." },";
  }

fclose($file);
?>

		]
	}]
});
chart.render();

}
</script>
</head>
<body>

<header>
<h2>ScorpioZero Neural Network Training.</h2>
</header>

<nav>
<p>
Welcome to the ScorpioZero training server. <br>
</p>

<p>
<a href="http://scorpiozero.ddns.net/users.php">List of users</a> <br>
<a href="http://scorpiozero.ddns.net/contrib.php">Contributions of users</a> <br>
<a href="http://scorpiozero.ddns.net/work.php">Active training runs</a> <br>
<a href="http://scorpiozero.ddns.net/scorpiozero">Networks, training games and data, matches etc</a> <br>
<a href="http://scorpiozero.ddns.net/matches.php">View matches between networks</a> <br>
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

</nav>

<article>
<div id="chartContainer" style="height: 570px; width: 870px;"></div>
</article>

<footer>
<p>--</p>
</footer>

<script src="https://canvasjs.com/assets/script/canvasjs.min.js"></script>

<style>
* {
  box-sizing: border-box;
}

body {
  font-family: Arial, Helvetica, sans-serif;
}

/* Style the header */
header {
  height: 5%;
  padding: 5px;
  background-color: #000;
  text-align: center;
  font-size: 15px;
  color: white;
}

/* Create two columns/boxes that floats next to each other */
nav {
  float: left;
  width: 25%;
  height: 654px;
  background: #ccc;
  padding: 20px;
}

/* Style the list inside the menu */
nav ul {
  list-style-type: none;
  padding: 0;
}

article {
  float: left;
  padding: 20px;
  width: 75%;
  background-color: #f1f1f1;
  height: 100%; /* only for demonstration, should be removed */
}

/* Clear floats after the columns */
section:after {
  content: "";
  display: table;
  clear: both;
}

/* Style the footer */
footer {
  background-color: #ccc;
  padding: 10px;
  text-align: center;
  color: white;
}

/* Responsive layout - makes the two columns/boxes stack on top of each other instead of next to each other, on small screens */
@media (max-width: 600px) {
  nav, article {
    width: 100%;
    height: auto;
  }
}
</style>
</body>
</html>
