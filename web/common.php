<?php
if( isset($_GET["runid"]) ) {
   $NET=$_GET["runid"];
} else {
   $dir = "scorpiozero/";
   $files = array_diff(scandir($dir), array('.', '..'));
   $files = array_reverse($files);
   $NET = "$dir$files[0]";
}
?>
