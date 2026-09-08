package com.example.arguslite.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

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
    MaterialTheme(colorScheme = ArgusScheme, typography = Typography,
        shapes = androidx.compose.material3.Shapes(
            small = androidx.compose.foundation.shape.RoundedCornerShape(14.dp),
            medium = androidx.compose.foundation.shape.RoundedCornerShape(20.dp),
            large = androidx.compose.foundation.shape.RoundedCornerShape(26.dp)),
        content = content)
}