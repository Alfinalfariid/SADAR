import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile


allprojects {
    repositories {
        google()
        mavenCentral()
    }
}


val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()

rootProject.layout.buildDirectory.value(
    newBuildDir
)


subprojects {

    val newSubprojectBuildDir: Directory =
        newBuildDir.dir(project.name)

    project.layout.buildDirectory.value(
        newSubprojectBuildDir
    )
}


// ============================================================
// FIX KHUSUS TFLITE_FLUTTER
// ============================================================
//
// tflite_flutter saat ini:
// Java target   = 11
//
// Jadi Kotlin milik plugin ini juga dibuat JVM 11.
//
// PENTING:
// Ini HANYA berlaku untuk plugin tflite_flutter.
// Aplikasi utama tetap Java/Kotlin 17.
// ============================================================

subprojects {

    if (project.name == "tflite_flutter") {

        tasks.withType<KotlinCompile>()
            .configureEach {

                compilerOptions {

                    jvmTarget.set(
                        JvmTarget.JVM_11
                    )
                }
            }
    }
}


subprojects {
    project.evaluationDependsOn(":app")
}


tasks.register<Delete>("clean") {

    delete(
        rootProject.layout.buildDirectory
    )
}