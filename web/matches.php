<!DOCTYPE html>
<html>
<body>

<head>  
<link rel="stylesheet" type="text/css" href="common.css">
</head>

<?php
include "navigation.php"
?>

<article>
<h2>Network evaluation matches</h2>

<?php
$dir = "$NET/matches/";
$files = array_diff(scandir($dir), array('.', '..'));

echo "<table style='border: solid 2px blue;'>\n";
echo "<tr><th>Matches</th></tr>";
echo "\t<tr>\n";
foreach( $files as $i=>$f ) {
   echo "\t\t<td style='width:300px;border:1px solid blue;'> <a href='games.php?match=$dir$f'>$f</a></td>\n";
   echo "\t</tr>\n";
}

echo "</table>\n";
?>
</article>

</body>
</html>

