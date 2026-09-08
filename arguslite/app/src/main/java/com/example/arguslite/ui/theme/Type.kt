package com.example.arguslite.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.example.arguslite.R

val Display = FontFamily(Font(R.font.fraunces))
val Body = FontFamily(Font(R.font.inter))
val Numbers = FontFamily(Font(R.font.spacemono))

val Typography = Typography(
    displaySmall = TextStyle(fontFamily = Display, fontWeight = FontWeight.Bold,
        fontSize = 40.sp, letterSpacing = 2.sp),
    headlineMedium = TextStyle(fontFamily = Display, fontWeight = FontWeight.SemiBold, fontSize = 26.sp),
    headlineSmall = TextStyle(fontFamily = Numbers, fontWeight = FontWeight.Bold, fontSize = 22.sp),
    titleLarge = TextStyle(fontFamily = Display, fontWeight = FontWeight.SemiBold, fontSize = 20.sp),
    titleMedium = TextStyle(fontFamily = Display, fontWeight = FontWeight.Medium, fontSize = 17.sp),
    titleSmall = TextStyle(fontFamily = Numbers, fontSize = 14.sp),
    bodyMedium = TextStyle(fontFamily = Body, fontSize = 15.sp, lineHeight = 22.sp),
    bodySmall = TextStyle(fontFamily = Body, fontSize = 12.5.sp, lineHeight = 18.sp),
    labelLarge = TextStyle(fontFamily = Numbers, fontSize = 14.sp),
)