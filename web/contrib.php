<!DOCTYPE html>
<html>
<body>

<h2>Game contirubtions of users</h2>

<?php

// Connecting, selecting database
$dbconn = pg_connect("host=localhost port=5432 dbname=scorpiozero user=postgres password=postgres")
    or die('Could not connect: ' . pg_last_error());

// Performing SQL query
//$query = 'SELECT * FROM contrib';
$query = 'select users.username,contrib.work_id,contrib.games from users inner join contrib on users.user_id = contrib.user_id';
$result = pg_query($query) or die('Query failed: ' . pg_last_error());

// Printing results in HTML
echo "<table style='border: solid 2px blue;'>\n";
echo "<tr><th>User</th><th>Training run</th><th>Number of games</th></tr>";
while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    echo "\t<tr>\n";
    foreach ($line as $col_value) {
        echo "\t\t<td style='width:300px;border:1px solid blue;'>$col_value</td>\n";
    }
    echo "\t</tr>\n";
}
echo "</table>\n";

// Free resultset
pg_free_result($result);

// Closing connection
pg_close($dbconn);

?>

</body>
</html>
