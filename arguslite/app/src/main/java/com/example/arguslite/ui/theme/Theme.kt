package com.example.arguslite.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val ArgusScheme = darkColorScheme(
    primary = ArgusTeal,
    onPrimary = ArgusNavy,
    secondary = ArgusCoral,
    onSecondary = ArgusNavy,
    background = ArgusNavy,
    onBackground = ArgusInk,
    surface = ArgusSurface,
    onSurface = ArgusInk,
    surfaceVariant = ArgusSurface,
    onSurfaceVariant = ArgusMuted,
)

@Composable
fun ARGUSLiteTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = ArgusScheme, typography = Typography, content = content)
}