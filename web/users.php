<!DOCTYPE html>
<html>
<body>

<h2>List of users</h2>

<?php

// Connecting, selecting database
$dbconn = pg_connect("host=localhost port=5432 dbname=scorpiozero user=postgres password=postgres")
    or die('Could not connect: ' . pg_last_error());

// Performing SQL query
$query = 'SELECT user_id,username,created_on,last_login FROM users';
$result = pg_query($query) or die('Query failed: ' . pg_last_error());

// Printing results in HTML
echo "<table style='border: solid 2px blue;'>\n";
echo "<tr><th>User ID</th><th>User name</th><th>Created on</th><th>Last login</th></tr>";
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
