# Downloading DataSet

- Download Dataset from here
[Open-Image](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/ "Open-Image")

-  store it like this
```
└── Open_Image
    ├── class-descriptions-boxable.csv
    ├── Test
    │   ├── test-annotations-bbox.csv
    │   ├── test-images.csv
    │   └── test.zip
    ├── Train
    │   ├── train_00.zip
    │   ├── train_01.zip
    │   ├── train_02.zip
    │   ├── train_03.zip
    │   ├── train_04.zip
    │   ├── train_05.zip
    │   ├── train_06.zip
    │   ├── train_07.zip
    │   ├── train_08.zip
    │   ├── train-annotations-bbox.csv
    │   └── train-images-boxable.csv
    └── Validation
        ├── validation-annotations-bbox.csv
        └── validation-images.csv
```

## install Basilia
- get package `git clone https://github.com/cna74/Basilia.git`
- go to  directory `cd Basilia`
- create a virtual enviornment `virtualenv -p ~/%your-python-path% /%venv dir%/` after replacing python path and venv dir it should be like this `virtualenv -p ~/anaconda3/bin/python3.6 ~/ElBasil`
- install requirements `pip install -r requierments.txt`

## in action
- open jupyter `jupyter lab` or `jupyter notebook`

<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<title>Demo</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    color: #000 !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.2.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.2.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.2.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.2.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.2.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.2.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=1);
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2);
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=3);
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1);
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1);
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
@media (max-width: 991px) {
  #ipython_notebook {
    margin-left: 10px;
  }
}
[dir="rtl"] #ipython_notebook {
  float: right !important;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#login_widget {
  float: right;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  text-align: center;
  vertical-align: middle;
  display: inline;
  opacity: 0;
  z-index: 2;
  width: 12ex;
  margin-right: -12ex;
}
.alternate_upload .btn-upload {
  height: 22px;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
[dir="rtl"] #tabs li {
  float: right;
}
ul#tabs {
  margin-bottom: 4px;
}
[dir="rtl"] ul#tabs {
  margin-right: 0px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons {
  float: left !important;
}
[dir="rtl"] .list_toolbar .pull-right {
  padding-top: 1px;
  float: left !important;
}
[dir="rtl"] .list_toolbar .pull-left {
  float: right !important;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: baseline;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
#tree-selector {
  padding-right: 0px;
}
[dir="rtl"] #tree-selector a {
  float: right;
}
#button-select-all {
  min-width: 50px;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
[dir="rtl"] #new-menu {
  text-align: right;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
[dir="rtl"] #running .col-sm-8 {
  float: right !important;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI colors. */
.ansibold {
  font-weight: bold;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size */
  /* .CodeMirror-lines */
  padding: 0;
  border: 0;
  border-radius: 0;
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul {
  list-style: disc;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ul ul {
  list-style: square;
  margin: 0em 2em;
}
.rendered_html ul ul ul {
  list-style: circle;
  margin: 0em 2em;
}
.rendered_html ol {
  list-style: decimal;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
  margin: 0em 2em;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  background-color: #fff;
  color: #000;
  font-size: 100%;
  padding: 0px;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: 1px solid black;
  border-collapse: collapse;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  border: 1px solid black;
  border-collapse: collapse;
  margin: 1em 2em;
}
.rendered_html td,
.rendered_html th {
  text-align: left;
  vertical-align: middle;
  padding: 4px;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget {
  float: right !important;
  float: right;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  margin-top: 6px;
}
span.save_widget span.filename {
  height: 1em;
  line-height: 1em;
  padding: 3px;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  display: none;
}
.command-shortcut:before {
  content: "(command)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">Basil</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">finder</span> <span class="o">=</span> <span class="n">Basil</span><span class="o">.</span><span class="n">Finder</span><span class="p">(</span><span class="n">subject</span><span class="o">=</span><span class="s1">&#39;lemon&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>images will export to data
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Find-them">Find them<a class="anchor-link" href="#Find-them">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">finder</span><span class="o">.</span><span class="n">fill_images_with_bbox</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>679 unique image
1756 unique object
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>train_01  100% (64 of 64) |##############| Elapsed Time: 0:04:35 Time:  0:04:35
train_02  100% (71 of 71) |##############| Elapsed Time: 0:05:06 Time:  0:05:06
train_00  100% (74 of 74) |##############| Elapsed Time: 0:05:16 Time:  0:05:16
train_03  100% (80 of 80) |##############| Elapsed Time: 0:05:41 Time:  0:05:41
train_04  100% (69 of 69) |##############| Elapsed Time: 0:04:34 Time:  0:04:34
train_07   72% (58 of 80) |##########    | Elapsed Time: 0:03:47 ETA:   0:01:25
train_06  100% (78 of 78) |##############| Elapsed Time: 0:04:54 Time:  0:04:54
train_07  100% (80 of 80) |##############| Elapsed Time: 0:04:51 Time:  0:04:51
train_08  100% (97 of 97) |##############| Elapsed Time: 0:02:53 Time:  0:02:53
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Test-a-few">Test a few<a class="anchor-link" href="#Test-a-few">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">finder</span><span class="o">.</span><span class="n">bbox_test</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Save-them">Save them<a class="anchor-link" href="#Save-them">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">finder</span><span class="o">.</span><span class="n">store_data</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Restore-them">Restore them<a class="anchor-link" href="#Restore-them">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">finder</span><span class="o">.</span><span class="n">restore</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>
