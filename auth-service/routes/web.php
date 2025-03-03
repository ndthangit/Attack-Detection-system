<?php

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {

    echo DB::table('employees')->get();
//    return view('welcome');
});
