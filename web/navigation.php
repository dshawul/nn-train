<!-- Left navigation panel --!>


<header>
<h2>ScorpioZero Chess Neural Network Training.</h2>
</header>

<nav>
<p>
Welcome to the ScorpioZero training server. <br>
</p>

<?php
include "common.php";
?>

<p>
Server status: <strong style="color: green;">ACTIVE</strong>  <br>
<!-- Server status: <strong style="color: red;">DOWN FOR MAINTENANCE</strong>  <br> --!>
</p>

<p>
<a href="http://scorpiozero.ddns.net/index.php">Elo graph</a> <br>
</p>

<p>
Statistics. <br>
</p>

<p>
<a href="http://scorpiozero.ddns.net/users.php">List of users</a> <br>
<a href="http://scorpiozero.ddns.net/contrib.php">Contributions of users</a> <br>
<a href="http://scorpiozero.ddns.net/work.php">Active training runs</a> <br>
</p>

<p>
Match games. <br>
</p>

<p>
<a href=<?php echo "http://scorpiozero.ddns.net/matches.php?runid=".$NET; ?> >View matches between networks</a> <br>
</p>

<p>
Networks and training data. <br>
</p>

<p>
<a href="http://scorpiozero.ddns.net/scorpiozero">Download networks and training data</a> <br>
</p>

<p>
Github pages. <br>
</p>

<a href="https://github.com/dshawul/nn-dist">Client for training</a> <br>
<a href="https://github.com/dshawul/nn-train">Server for training</a> <br>
<a href="https://github.com/dshawul/nn-probe">Probing code</a> <br>

<p>
Live help and discussion. <br>
</p>

<a href="https://discord.gg/BPYhwG">Discord channel</a> <br>

<p>
Discussion. <br>
</p>

<a href="https://groups.google.com/forum/#!forum/scorpiozero">Discussion forum</a> <br>

<p>
Tensorboard graph. <br>
</p>

<a href="https://scorpiozero.ddns.net:6006">Tensorboard</a> <br>

</nav>
