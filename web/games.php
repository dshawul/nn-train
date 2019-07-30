<HTML>

<HEAD>
    <TITLE>ScorpioZero training games</TITLE>
</HEAD>

<BODY> 

<SCRIPT LANGUAGE="JavaScript">
</SCRIPT>

<H2> ScorpioZero training games<br>
<font size="2">
(courtesy of <a target="v" href="http://chesstempo.com">Chess Tempo)</a>
</H2>

<font size="3">

<!-- Support libraries from Yahoo YUI project -->
<script type="text/javascript"

    src="http://chesstempo.com/js/pgnyui.js">

</script>


<script type="text/javascript"

    src="http://chesstempo.com/js/pgnviewer.js">

</script>


<link
   type="text/css"
   rel="stylesheet"
   href="http://chesstempo.com/css/board-min.css">
</link>


<p id="cmatch"><?php echo $_GET["match"]  ?></p>

<script>
 var pgnf = document.getElementById('cmatch').innerHTML;
 new PgnViewer(
   { 
       boardName: "ed",
       showCoordinates: true,
       pgnFile: pgnf,
       pieceSet: 'merida',
       pieceSize: 65
   }
 );
</script>


<style type="text/css"/>

.ct-nav-buttons{padding-bottom:15px;padding-top:15px;}
.ct-back,
.ct-forward,
.ct-start,
.ct-end,
.ct-play,
.ct-stop{vertical-align:middle;}
.ct-mainline-commentary{padding-left:15px;}


</style>

<div id="ed-container"></div>

<div id="ed-moves"></div>

</BODY>
</HTML>
