plugins {
    id("com.android.application")

    // Flutter Gradle Plugin harus setelah Android plugin.
    id("dev.flutter.flutter-gradle-plugin")
}


android {

    namespace = "com.example.mobile"

    compileSdk =
        flutter.compileSdkVersion

    ndkVersion =
        flutter.ndkVersion


    // ========================================================
    // JAVA APP = 17
    // ========================================================

    compileOptions {

        sourceCompatibility =
            JavaVersion.VERSION_17

        targetCompatibility =
            JavaVersion.VERSION_17
    }


    defaultConfig {

        applicationId =
            "com.example.mobile"

        minSdk =
            flutter.minSdkVersion

        targetSdk =
            flutter.targetSdkVersion

        versionCode =
            flutter.versionCode

        versionName =
            flutter.versionName
    }


    buildTypes {

        release {

            signingConfig =
                signingConfigs
                    .getByName("debug")
        }
    }
}


// ============================================================
// KOTLIN APP = JVM 17
// ============================================================

kotlin {

    compilerOptions {

        jvmTarget =
            org.jetbrains.kotlin.gradle.dsl
                .JvmTarget.JVM_17
    }
}


flutter {

    source = "../.."
}