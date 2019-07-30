<html>
<body>
<h2>Netowrk evaluation matches</h2>

<?php
$dir = "scorpiozero/nets-1/matches/";
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
</body>
</html>

